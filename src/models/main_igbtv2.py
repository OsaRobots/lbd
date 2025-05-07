from functools import partial
from flax import nnx
import flax.linen as fnn
import jax
from jax import lax
from typing import Dict, Tuple, Sequence
from jax import random, numpy as jnp
import optax
import matplotlib.pyplot as plt
import ott.neural.networks.icnn as ott_icnn
import pandas as pd

# helpers
def dirichlet_log_prob(theta: jax.Array, alpha_dist_params: jax.Array) -> jax.Array:
    if theta.ndim == 1:
        return jax.scipy.stats.dirichlet.logpdf(theta, alpha_dist_params)
    else:
        return jax.scipy.stats.dirichlet.logpdf(theta.T, alpha_dist_params)

def sample_uniform_on_simplex(key: jax.random.PRNGKey, K_dims: int, num_samples: int) -> jax.Array:
    return jax.random.dirichlet(key, jnp.ones(K_dims), shape=(num_samples,))

class IGBTNet(nnx.Module):
    def __init__(self, in_dims_x0: int, hidden_dims: int, theta_dims: int, *, rngs: nnx.Rngs):
        self.cost_net = nnx.Sequential(
            nnx.Linear(in_dims_x0 + theta_dims, hidden_dims, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dims, hidden_dims, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dims, 1, rngs=rngs)
        )
        self.alpha_scalar = nnx.Param(jax.random.normal(rngs.params(), ()))

        self.beta_potential = nnx.Sequential(
            nnx.Linear(theta_dims, hidden_dims, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dims, hidden_dims, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dims, 1, rngs=rngs)
        )
        
        self._eps_x_log = nnx.Param(jnp.array(jnp.log(1.0)))
        self._eps_gamma_log = nnx.Param(jnp.array(jnp.log(1.0)))
        self._eps_theta_log = nnx.Param(jnp.array(jnp.log(1.1)))

    @property
    def eps_x(self): return jnp.exp(self._eps_x_log.value)
    @property
    def eps_gamma(self): return jnp.exp(self._eps_gamma_log.value)
    @property
    def eps_theta(self): return jnp.exp(self._eps_theta_log.value)

    def cost(self, x0: jax.Array, theta: jax.Array) -> jax.Array:
        if x0.ndim < theta.ndim and theta.ndim == x0.ndim + 1 :
             x0_expanded = jnp.expand_dims(x0, axis=-2)
        elif x0.ndim == theta.ndim and x0.shape[:-1] == theta.shape[:-1]:
             x0_expanded = x0
        else:
             if x0.ndim == 1 and theta.ndim == 2:
                 x0_expanded = x0[None, :]
             else:
                 x0_expanded = x0
        x0_broadcasted = jnp.broadcast_to(x0_expanded, (*theta.shape[:-1], x0.shape[-1]))
        x_theta = jnp.concatenate([x0_broadcasted, theta], axis=-1)
        return jnp.squeeze(self.cost_net(x_theta), axis=-1)

    def m_tilde_log_space(self, x0: jax.Array, theta: jax.Array, log_rho_theta: jax.Array) -> jax.Array:
        k = self.eps_theta - self.eps_gamma
        k_stable = jnp.maximum(k, 1e-6)
        g_theta = self.beta_potential(theta)
        if g_theta.ndim > 0 and g_theta.shape[-1] == 1 and log_rho_theta.ndim == g_theta.ndim -1 : # common if ICNN outputs [...,1]
            g_theta = jnp.squeeze(g_theta, axis=-1)
        c_w_x0_theta = self.cost(x0, theta)
        log_m_tilde_payload = (self.eps_theta * log_rho_theta - g_theta - c_w_x0_theta) / k_stable - 1.0
        return log_m_tilde_payload

    def m_tilde(self, x0: jax.Array, theta: jax.Array, log_rho_theta: jax.Array) -> jax.Array:
        return jnp.exp(self.m_tilde_log_space(x0, theta, log_rho_theta))

    def M_f_val(self) -> jax.Array:
        return jnp.exp(self.alpha_scalar.value / jnp.maximum(self.eps_x, 1e-6) - 1.0)

    def M_integral_mc(self, x0_t: jax.Array, sampled_thetas: jax.Array, log_rho_theta_at_samples: jax.Array) -> jax.Array:
        m_tilde_values = self.m_tilde(x0_t, sampled_thetas, log_rho_theta_at_samples)
        return jnp.mean(m_tilde_values, axis=0 if m_tilde_values.ndim > 0 else None)

    def E_pt_theta_mc(self, x0_t: jax.Array, sampled_thetas: jax.Array, log_rho_theta_at_samples: jax.Array, M_f_current: jax.Array) -> jax.Array:
        m_tilde_values = self.m_tilde(x0_t, sampled_thetas, log_rho_theta_at_samples)
        integral_m_theta = jnp.mean(m_tilde_values[:, None] * sampled_thetas, axis=0)
        return integral_m_theta / jnp.maximum(M_f_current, 1e-6)

# --- loss_Jt_fn (Updated for one-hot human_action_t) ---
def loss_Jt_fn(model: IGBTNet,
               x0_t: jax.Array,
               alpha_dirichlet_t: jax.Array,
               human_action_t_one_hot: jax.Array, # Now expects one-hot JAX array
               key_mc_sampling: jax.random.PRNGKey,
               num_theta_samples: int,
               lambda_reg_cost_net: float,
               theta_dims: int
               ):
    k_val = model.eps_theta - model.eps_gamma
    k_val_stable = jnp.maximum(k_val, 1e-6)

    sampled_thetas = sample_uniform_on_simplex(key_mc_sampling, theta_dims, num_theta_samples)
    log_rho_theta_at_samples = dirichlet_log_prob(sampled_thetas, alpha_dirichlet_t)

    M_integral_val = model.M_integral_mc(x0_t, sampled_thetas, log_rho_theta_at_samples)
    M_f_val = model.M_f_val()
    E_p_t_theta_val = model.E_pt_theta_mc(x0_t, sampled_thetas, log_rho_theta_at_samples, M_f_val) # [K]

    # calculate probability of human action using one-hot encoding
    # E_p_t_theta_val is [K], human_action_t_one_hot is [K]
    prob_human_action = jnp.sum(E_p_t_theta_val * human_action_t_one_hot)
    nll_term = -jnp.log(jnp.maximum(prob_human_action, 1e-9))

    # R_w_term_explicit = 0.0
    # if lambda_reg_cost_net > 0:
    #     cost_net_params_state, _ = nnx.split(model.cost_net, nnx.Param)
    #     for var_state in jax.tree_util.tree_leaves(cost_net_params_state):
    #         if isinstance(var_state, nnx.Param):
    #              R_w_term_explicit += 0.5 * jnp.sum(var_state.value**2)

    # J_t_val = nll_term + lambda_reg_cost_net * R_w_term_explicit + \
    #           k_val_stable * M_integral_val + model.eps_x * M_f_val

    J_t_val = nll_term + k_val_stable * M_integral_val + model.eps_x * M_f_val
    aux_data = {'E_pt_theta': E_p_t_theta_val, 'nll': nll_term}
    return J_t_val, aux_data

def create_optimizer_states(model: IGBTNet, learning_rate: float, weight_decay: float):
    # def w_actual_param_filter(path: tuple[str,...], node_value):
    #     return isinstance(node_value, nnx.Param) and is_w_param(path, node_value)
    # def fg_actual_param_filter(path: tuple[str,...], node_value):
    #     return isinstance(node_value, nnx.Param) and is_fg_param(path, node_value)

    # w_params_module_state, remaining_after_w = nnx.split(model, w_actual_param_filter)
    # optimizer_w = optax.adamw(learning_rate_w, weight_decay=weight_decay_w)
    # opt_state_w = optimizer_w.init(w_params_module_state)

    # fg_params_module_state, _ = nnx.split(remaining_after_w, fg_actual_param_filter)
    # optimizer_fg = optax.adamw(learning_rate_fg)
    # opt_state_fg = optimizer_fg.init(fg_params_module_state)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, weight_decay=weight_decay))
    return optimizer

@partial(nnx.jit, static_argnums=(6, 7, 8))
def train_step_igbt(
    model: IGBTNet,
    optimizer: nnx.Optimizer, 
    current_alpha_dirichlet: jax.Array,
    x0_t: jax.Array,
    human_action_t_one_hot: jax.Array, # Now expects one-hot JAX array
    key_mc_sampling: jax.random.PRNGKey,
    num_theta_samples: int,
    lambda_reg_cost_net: float,
    theta_dims: int
):
    grad_fn_Jt = nnx.value_and_grad(loss_Jt_fn, argnums=0, has_aux=True)
    (Jt_value, aux_data), grads = grad_fn_Jt(
        model, x0_t, current_alpha_dirichlet, human_action_t_one_hot, # Pass one-hot directly
        key_mc_sampling, num_theta_samples, lambda_reg_cost_net, theta_dims
    )

    optimizer.update(grads)
    E_pt_theta_val = aux_data['E_pt_theta']
    new_alpha_dirichlet = current_alpha_dirichlet + E_pt_theta_val
    new_alpha_dirichlet = jnp.maximum(new_alpha_dirichlet, 1e-6)

    return model, new_alpha_dirichlet, Jt_value, aux_data['nll']

# --- train_igbt_model (main loop, updated for one-hot human_action_t) ---
def train_igbt_model(
    num_rounds: int,
    get_round_data_fn, # signature: (round_idx) -> (x0_t_np, human_action_t_one_hot_np)
    in_dims_x0: int,
    hidden_dims: int,
    theta_dims: int,
    initial_alpha_dirichlet_params: jax.Array,
    num_theta_samples: int,
    lambda_reg_cost_net: float,
    learning_rate: float,
    weight_decay: float,
    seed: int = 0
):
    key_master = jax.random.PRNGKey(seed)
    key_init_model, key_training_loop = jax.random.split(key_master)
    model_rngs = nnx.Rngs(params=key_init_model, dropout=jax.random.key(seed + 1))
    model_def = lambda rngs_payload: IGBTNet(in_dims_x0=in_dims_x0, hidden_dims=hidden_dims, theta_dims=theta_dims, rngs=rngs_payload)
    model = model_def(model_rngs)
    optimizer = create_optimizer_states(model, learning_rate, weight_decay)
    current_alpha_dirichlet = initial_alpha_dirichlet_params
    Jt_history, nll_history, alpha_sum_history = [], [], []

    print(f"Starting IGBT training for {num_rounds} rounds...")
    for t in range(num_rounds):
        key_round_mc_sampling, key_training_loop = jax.random.split(key_training_loop)
        x0_t_np, human_action_t_one_hot_np = get_round_data_fn(t) 
        x0_t_jax = jnp.asarray(x0_t_np, dtype=jnp.float32)
        human_action_t_one_hot_jax = jnp.asarray(human_action_t_one_hot_np, dtype=jnp.float32) 

        model, current_alpha_dirichlet, Jt_val, nll_val = train_step_igbt(model, 
                                                                          optimizer, 
                                                                          current_alpha_dirichlet, 
                                                                          x0_t_jax, 
                                                                          human_action_t_one_hot_jax, 
                                                                          key_round_mc_sampling, 
                                                                          num_theta_samples, 
                                                                          lambda_reg_cost_net, 
                                                                          theta_dims)
        Jt_history.append(float(Jt_val))
        nll_history.append(float(nll_val))
        alpha_sum_history.append(float(jnp.sum(current_alpha_dirichlet)))
        if t % 100 == 0 or t == num_rounds - 1:
            print(f"Round {t}/{num_rounds}: J_t = {Jt_val:.4f}, NLL = {nll_val:.4f}, Sum(alpha_t+1) = {alpha_sum_history[-1]:.2f}")

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1); plt.plot(Jt_history); plt.title("$J_t$ Objective"); plt.xlabel("Round"); plt.grid(True)
    plt.subplot(1, 3, 2); plt.plot(nll_history); plt.title("NLL Term"); plt.xlabel("Round"); plt.grid(True)
    plt.subplot(1, 3, 3); plt.plot(alpha_sum_history); plt.title("Sum of $\\alpha_t$"); plt.xlabel("Round"); plt.grid(True)
    plt.tight_layout(); plt.show()
    return model, Jt_history, nll_history, alpha_sum_history

if __name__ == '__main__':
    THETA_DIMS_K_example = 3
    IN_DIMS_X0_example = 5

    def get_dummy_round_data(round_idx):
        data_key = jax.random.key(round_idx)
        action_key = jax.random.key(round_idx + 10000)
        dummy_x0_np = jax.random.normal(data_key, (IN_DIMS_X0_example,))
        dummy_action_idx_np = jax.random.randint(action_key, (), 0, THETA_DIMS_K_example)
        dummy_action_one_hot_np = jax.nn.one_hot(dummy_action_idx_np, THETA_DIMS_K_example, dtype=jnp.float32)
        return dummy_x0_np, dummy_action_one_hot_np 

    train_igbt_model(
        num_rounds=100,
        get_round_data_fn=get_dummy_round_data,
        in_dims_x0=IN_DIMS_X0_example,
        hidden_dims=32,
        theta_dims=THETA_DIMS_K_example,
        initial_alpha_dirichlet_params=jnp.ones(THETA_DIMS_K_example) * 0.5,
        num_theta_samples=64,
        lambda_reg_cost_net=0.0,
        learning_rate=1e-4,
        weight_decay=1e-3,
        seed=42
    )