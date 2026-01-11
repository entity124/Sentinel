import pandas as pd
import numpy as np

class TransactionAnalyzer:
    def __init__(self, raw_data_path):
        self.df = pd.read_csv(raw_data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
    def get_features_for_model(self):
        """
        Transforms raw logs into the exact format the Governance Model expects.
        """
        # 1. Sort by User and Merchant to find patterns
        self.df = self.df.sort_values(by=['user_id', 'merchant', 'timestamp'])

        # group by user+merchant to find price change %
        self.df['prev_amount'] = self.df.groupby(['user_id', 'merchant'])['amount'].shift(1)
        self.df['price_change'] = (self.df['amount'] - self.df['prev_amount']) / self.df['prev_amount']
        
        self.df['price_change'] = self.df['price_change'].fillna(0)
        
        # Calculate Frequency and map to monthly/weekly etc.
        self.df['prev_date'] = self.df.groupby(['user_id', 'merchant'])['timestamp'].shift(1)
        self.df['days_since_last'] = (self.df['timestamp'] - self.df['prev_date']).dt.days
        self.df['frequency_inferred'] = self.df['days_since_last'].apply(self._infer_freq)

        #assigns income bracket to users. Would actually come from bank in real implementation
        np.random.seed(42)
        users = self.df['user_id'].unique()
        income_map = {u: np.random.choice(['High', 'Low']) for u in users}
        self.df['income_bracket'] = self.df['user_id'].map(income_map)

        # select columns for model input
        model_input = self.df[[
            'transaction_id', 
            'description', 
            'amount', 
            'frequency_inferred', 
            'price_change', 
            'income_bracket', 
            'category'
        ]].copy()
        
        # Rename to match Governance Sheet 
        model_input.columns = [
            'Transaction_ID', 'Description', 'Amount', 'Frequency', 
            'Price_Change_Pct', 'Income_Bracket', 'Category'
        ]
        
        return model_input

    def _infer_freq(self, days):
        if pd.isna(days): return "First_Txn"
        if 25 <= days <= 35: return "Monthly"
        if 6 <= days <= 8: return "Weekly"
        if 360 <= days <= 370: return "Annually"
        return "Irregular"

# --- TEST BLOCK ---
if __name__ == "__main__":
    analyzer = TransactionAnalyzer('mock_transactions.csv')
    clean_data = analyzer.get_features_for_model()
    
    #output data to new csv. This is the data to feed to granite
    clean_data.to_csv('live_model_input.csv', index=False)