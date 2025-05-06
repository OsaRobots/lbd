import pandas as pd
import matplotlib.pyplot as plt

# 1) Load each model’s accuracy CSV
igbt_df         = pd.read_csv('igbt_accuracy.csv')
lstm_df         = pd.read_csv('lstm_accuracy.csv')
transformer_df  = pd.read_csv('transformer_accuracy.csv')

# 2) Plot all on one figure
plt.figure(figsize=(10, 6))

# IGBT
plt.plot(igbt_df['epoch'], igbt_df['train_accuracy'],   label='IGBT Train')
plt.plot(igbt_df['epoch'], igbt_df['test_accuracy'],    label='IGBT Test',    linestyle='--')

# LSTM
plt.plot(lstm_df['epoch'], lstm_df['train_accuracy'],   label='LSTM Train')
plt.plot(lstm_df['epoch'], lstm_df['test_accuracy'],    label='LSTM Test',    linestyle='--')

# Transformer
plt.plot(transformer_df['epoch'], transformer_df['train_accuracy'],  label='Transformer Train')
plt.plot(transformer_df['epoch'], transformer_df['test_accuracy'],   label='Transformer Test', linestyle='--')

# 3) Formatting
plt.title('Train/Test Accuracy Comparison: IGBT, LSTM, Transformer')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()

# 4) Show (or save) the plot
plt.savefig('all_models_accuracy.png', dpi=150)

# plt.show()
