import pandas as pd

# Load the prediction files
preds_user = pd.read_csv("Gourangi_predictions.csv")
preds_friend = pd.read_csv("Dipanwita_Rano_HW4.csv")

# Check if the number of rows is the same
if len(preds_user) != len(preds_friend):
    print("Error: The files have a different number of rows.")
else:
    # Compare the predictions
    comparison = preds_user['pred_cnn'] != preds_friend['pred_cnn']
    
    # Calculate the number of differing rows
    differing_rows = comparison.sum()
    total_rows = len(preds_user)
    
    print(f"Total number of rows: {total_rows}")
    print(f"Number of differing rows: {differing_rows}")
