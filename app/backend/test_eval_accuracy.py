"""
Quick verification that evaluation accuracy logic is correct.
Run: cd app/backend && python test_eval_accuracy.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Simulate confusion matrix and verify formulas
def verify_metrics():
    # Example: 100 malicious samples, 80 correctly blocked (TP), 20 missed (FN)
    tp, tn, fp, fn = 80, 0, 0, 20
    total = tp + tn + fp + fn
    assert total == 100, "Total should be 100"
    
    accuracy = (tp + tn) / total * 100
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
    fpr = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("=== Malicious-only dataset (e.g. 100 attacks) ===")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Accuracy = (TP+TN)/Total = {accuracy:.2f}%  (correct: 80%)")
    print(f"  Recall = TP/(TP+FN) = {recall:.2f}%  (correct: 80%)")
    print(f"  FPR = FP/(FP+TN) = {fpr:.2f}%  (correct: N/A when TN=0)")
    assert abs(accuracy - 80) < 0.01
    assert abs(recall - 80) < 0.01
    print("  [OK] Malicious metrics\n")
    
    # Example: 100 benign samples, 90 correctly allowed (TN), 10 false alarms (FP)
    tp, tn, fp, fn = 0, 90, 10, 0
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total * 100
    fpr = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
    print("=== Benign-only dataset (e.g. 100 benign) ===")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Accuracy = (TP+TN)/Total = {accuracy:.2f}%  (correct: 90%)")
    print(f"  FPR = FP/(FP+TN) = {fpr:.2f}%  (correct: 10%)")
    assert abs(accuracy - 90) < 0.01
    assert abs(fpr - 10) < 0.01
    print("  [OK] Benign metrics\n")
    
    # Mixed: 50 benign + 50 malicious
    tp, tn, fp, fn = 45, 48, 2, 5
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total * 100
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
    fpr = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print("=== Mixed dataset (50 benign + 50 malicious) ===")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Accuracy = {accuracy:.2f}%  (correct: 93%)")
    print(f"  Recall = {recall:.2f}%, FPR = {fpr:.2f}%")
    assert abs(accuracy - 93) < 0.01
    print("  [OK] Mixed metrics\n")
    
    print("All accuracy formula checks passed!")

if __name__ == "__main__":
    verify_metrics()
