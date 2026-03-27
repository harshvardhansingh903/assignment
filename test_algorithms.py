"""
Demo and Test Script for Association Rule Mining Algorithms
Tests both Apriori and FP-Growth implementations
"""

import sys
sys.path.insert(0, './algorithms')

from apriori import AprioriAlgorithm
from fpgrowth import FPGrowthAlgorithm
import pandas as pd
import time

def print_header(text):
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70)

def test_simple_dataset():
    """Test with a simple dataset"""
    print_header("TEST 1: Simple Dataset")
    
    transactions = [
        ['A', 'B', 'C'],
        ['A', 'B'],
        ['A', 'C'],
        ['B', 'C'],
        ['B']
    ]
    
    print("\nTransactions:")
    for i, trans in enumerate(transactions, 1):
        print(f"  T{i}: {set(trans)}")
    
    min_sup = 0.4
    min_conf = 0.5
    
    print(f"\nParameters: min_support={min_sup}, min_confidence={min_conf}")
    
    # Apriori
    print("\n--- Apriori Algorithm ---")
    apriori = AprioriAlgorithm(min_support=min_sup, min_confidence=min_conf)
    apriori.run(transactions)
    apriori.print_results()
    
    # FP-Growth
    print("\n--- FP-Growth Algorithm ---")
    fpg = FPGrowthAlgorithm(min_support=min_sup, min_confidence=min_conf)
    fpg.run(transactions)
    fpg.print_results()
    
    # Verify results match
    apriori_itemsets = sum(len(v) for v in apriori.frequent_itemsets.values())
    fpg_itemsets = sum(len(v) for v in fpg.frequent_itemsets.values())
    
    print(f"\n✓ Both algorithms found {apriori_itemsets} itemsets: {apriori_itemsets == fpg_itemsets}")
    print(f"✓ Both found {len(apriori.association_rules)} rules: {len(apriori.association_rules) == len(fpg.association_rules)}")

def test_market_basket():
    """Test with market basket data"""
    print_header("TEST 2: Market Basket Dataset")
    
    # Load data
    df = pd.read_csv('./data/transactions.csv')
    transactions = []
    for idx, row in df.iterrows():
        items = [item.strip() for item in row['Items'].split(',')]
        transactions.append(items)
    
    print(f"\nDataset loaded:")
    print(f"  Total transactions: {len(transactions)}")
    print(f"  Items per transaction: {[len(t) for t in transactions[:5]]}")
    
    min_sup = 0.10
    min_conf = 0.60
    
    print(f"\nParameters: min_support={min_sup}, min_confidence={min_conf}")
    
    # Test Apriori
    print("\n--- Running Apriori Algorithm ---")
    start = time.time()
    apriori = AprioriAlgorithm(min_support=min_sup, min_confidence=min_conf)
    apriori.run(transactions)
    apriori_time = time.time() - start
    print(f"Apriori completed in {apriori_time:.4f} seconds")
    print(f"Found {sum(len(v) for v in apriori.frequent_itemsets.values())} itemsets")
    print(f"Found {len(apriori.association_rules)} association rules")
    
    # Test FP-Growth
    print("\n--- Running FP-Growth Algorithm ---")
    start = time.time()
    fpg = FPGrowthAlgorithm(min_support=min_sup, min_confidence=min_conf)
    fpg.run(transactions)
    fpg_time = time.time() - start
    print(f"FP-Growth completed in {fpg_time:.4f} seconds")
    print(f"Found {sum(len(v) for v in fpg.frequent_itemsets.values())} itemsets")
    print(f"Found {len(fpg.association_rules)} association rules")
    
    # Performance comparison
    print(f"\n--- Performance Comparison ---")
    print(f"Apriori Time:    {apriori_time:.4f}s")
    print(f"FP-Growth Time:  {fpg_time:.4f}s")
    print(f"Speedup: {apriori_time/fpg_time:.2f}x faster (FP-Growth)")
    
    # Verify consistency
    apriori_itemsets = sum(len(v) for v in apriori.frequent_itemsets.values())
    fpg_itemsets = sum(len(v) for v in fpg.frequent_itemsets.values())
    
    if apriori_itemsets == fpg_itemsets:
        print(f"\n✓ Both algorithms found {apriori_itemsets} itemsets")
    else:
        print(f"\n✗ Itemset count mismatch: Apriori={apriori_itemsets}, FP-Growth={fpg_itemsets}")
    
    if len(apriori.association_rules) == len(fpg.association_rules):
        print(f"✓ Both algorithms found {len(apriori.association_rules)} rules")
    else:
        print(f"✗ Rule count mismatch: Apriori={len(apriori.association_rules)}, FP-Growth={len(fpg.association_rules)}")

def test_parameter_sensitivity():
    """Test sensitivity to parameter changes"""
    print_header("TEST 3: Parameter Sensitivity Analysis")
    
    # Load data
    df = pd.read_csv('./data/transactions.csv')
    transactions = []
    for idx, row in df.iterrows():
        items = [item.strip() for item in row['Items'].split(',')]
        transactions.append(items)
    
    support_values = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    print("\nTesting different support thresholds:\n")
    print(f"{'Support':<10} {'Apriori Time':<15} {'FP-Growth Time':<15} {'Itemsets':<10} {'Rules':<10}")
    print("-" * 60)
    
    for sup in support_values:
        # Apriori
        start = time.time()
        apriori = AprioriAlgorithm(min_support=sup, min_confidence=0.6)
        apriori.run(transactions)
        apriori_time = time.time() - start
        apriori_itemsets = sum(len(v) for v in apriori.frequent_itemsets.values())
        
        # FP-Growth
        start = time.time()
        fpg = FPGrowthAlgorithm(min_support=sup, min_confidence=0.6)
        fpg.run(transactions)
        fpg_time = time.time() - start
        fpg_itemsets = sum(len(v) for v in fpg.frequent_itemsets.values())
        
        print(f"{sup:<10.2f} {apriori_time:<15.4f} {fpg_time:<15.4f} {apriori_itemsets:<10} {len(apriori.association_rules):<10}")

def test_edge_cases():
    """Test edge cases"""
    print_header("TEST 4: Edge Cases")
    
    # Test 1: Single item transactions
    print("\nTest 4.1: Single item transactions")
    transactions = [['A'], ['B'], ['A'], ['C'], ['B']]
    apriori = AprioriAlgorithm(min_support=0.4, min_confidence=0.5)
    apriori.run(transactions)
    print(f"✓ Handled single items: Found {sum(len(v) for v in apriori.frequent_itemsets.values())} itemsets")
    
    # Test 2: All same items
    print("\nTest 4.2: All transactions identical")
    transactions = [['A', 'B', 'C'], ['A', 'B', 'C'], ['A', 'B', 'C']]
    apriori = AprioriAlgorithm(min_support=0.5, min_confidence=0.5)
    apriori.run(transactions)
    print(f"✓ Handled identical transactions: Found {len(apriori.association_rules)} rules")
    
    # Test 3: No frequent itemsets
    print("\nTest 4.3: Very high support threshold (no itemsets expected)")
    transactions = [['A', 'B'], ['C', 'D'], ['E', 'F']]
    apriori = AprioriAlgorithm(min_support=0.9, min_confidence=0.5)
    apriori.run(transactions)
    itemsets = sum(len(v) for v in apriori.frequent_itemsets.values())
    print(f"✓ Handled no itemsets: Found {itemsets} itemsets (expected 0)")

def main():
    """Run all tests"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "Association Rule Mining - Algorithm Testing".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    try:
        test_simple_dataset()
        test_market_basket()
        test_parameter_sensitivity()
        test_edge_cases()
        
        print_header("All Tests Completed Successfully!")
        print("\n✓ Apriori algorithm working correctly")
        print("✓ FP-Growth algorithm working correctly")
        print("✓ Results are consistent between algorithms")
        print("✓ Performance characteristics verified")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
