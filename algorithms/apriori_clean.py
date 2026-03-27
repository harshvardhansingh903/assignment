from itertools import combinations
from collections import defaultdict, Counter
import pandas as pd


class AprioriAlgorithm:
    def __init__(self, min_support=0.1, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transactions = []
        self.num_transactions = 0
        self.frequent_itemsets = {}
        self.association_rules = []

    def load_transactions(self, transaction_list):
        self.transactions = [set(trans) for trans in transaction_list]
        self.num_transactions = len(self.transactions)

    def calculate_support(self, itemset):
        count = sum(1 for trans in self.transactions if itemset.issubset(trans))
        return count / self.num_transactions

    def get_candidates_1(self):
        items = defaultdict(int)
        for trans in self.transactions:
            for item in trans:
                items[item] += 1

        frequent_1 = {}
        for item, count in items.items():
            support = count / self.num_transactions
            if support >= self.min_support:
                frequent_1[frozenset([item])] = support

        return frequent_1

    def get_candidates_k(self, prev_frequent, k):
        items_list = list(prev_frequent.keys())
        candidates = []

        for i in range(len(items_list)):
            for j in range(i + 1, len(items_list)):
                union = items_list[i] | items_list[j]
                if len(union) == k:
                    candidates.append(union)

        candidates = list(set(candidates))

        frequent_k = {}
        for candidate in candidates:
            support = self.calculate_support(candidate)
            if support >= self.min_support:
                frequent_k[candidate] = support

        return frequent_k

    def find_frequent_itemsets(self):
        self.frequent_itemsets = {}

        frequent_1 = self.get_candidates_1()
        if not frequent_1:
            print("No frequent itemsets found with given support threshold")
            return {}

        self.frequent_itemsets[1] = frequent_1
        previous_frequent = frequent_1
        k = 2

        while previous_frequent:
            frequent_k = self.get_candidates_k(previous_frequent, k)
            if not frequent_k:
                break
            self.frequent_itemsets[k] = frequent_k
            previous_frequent = frequent_k
            k += 1

        return self.frequent_itemsets

    def generate_rules(self):
        self.association_rules = []
        
        for k in self.frequent_itemsets:
            if k >= 2:
                for itemset, support in self.frequent_itemsets[k].items():
                    itemset_list = list(itemset)
                    
                    for antecedent_size in range(1, k):
                        for antecedent in combinations(itemset_list, antecedent_size):
                            antecedent = frozenset(antecedent)
                            consequent = itemset - antecedent
                            
                            antecedent_support = self.frequent_itemsets[antecedent_size][antecedent]
                            confidence = support / antecedent_support
                            
                            if confidence >= self.min_confidence:
                                consequent_itemset = frozenset(consequent)
                                consequent_size = len(consequent)
                                consequent_support = self.frequent_itemsets[consequent_size][consequent_itemset]
                                lift = confidence / consequent_support
                                
                                rule = {
                                    'antecedent': antecedent,
                                    'consequent': consequent_itemset,
                                    'support': support,
                                    'confidence': confidence,
                                    'lift': lift
                                }
                                self.association_rules.append(rule)

        self.association_rules.sort(key=lambda x: (-x['confidence'], -x['lift']))
        
        return self.association_rules

    def run(self, transaction_list):
        self.load_transactions(transaction_list)
        self.find_frequent_itemsets()
        self.generate_rules()
        
        return self.frequent_itemsets, self.association_rules

    def print_results(self):
        print("=" * 80)
        print("APRIORI ALGORITHM RESULTS")
        print("=" * 80)
        print(f"\nMin Support: {self.min_support}")
        print(f"Min Confidence: {self.min_confidence}")
        print(f"Total Transactions: {self.num_transactions}\n")

        print("FREQUENT ITEMSETS:")
        print("-" * 80)
        for k in sorted(self.frequent_itemsets.keys()):
            print(f"\n{k}-Itemsets:")
            for itemset, support in sorted(self.frequent_itemsets[k].items(), 
                                         key=lambda x: -x[1]):
                itemset_str = str(set(itemset)).ljust(30)
                print(f"  {itemset_str} -> Support: {support:.4f}")

        print(f"\n\nASSOCIATION RULES: (Found {len(self.association_rules)} rules)")
        print("-" * 80)
        for i, rule in enumerate(self.association_rules[:10], 1):
            antecedent = ', '.join(sorted(rule['antecedent']))
            consequent = ', '.join(sorted(rule['consequent']))
            print(f"{i}. {antecedent} => {consequent}")
            print(f"   Support: {rule['support']:.4f}, Confidence: {rule['confidence']:.4f}, Lift: {rule['lift']:.4f}\n")

        if len(self.association_rules) > 10:
            print(f"... and {len(self.association_rules) - 10} more rules")


if __name__ == "__main__":
    transactions = [
        ['Bread', 'Milk', 'Beer', 'Diapers'],
        ['Bread', 'Diapers', 'Beer', 'Eggs'],
        ['Milk', 'Diapers', 'Beer', 'Cola'],
    ]

    apriori = AprioriAlgorithm(min_support=0.1, min_confidence=0.6)
    apriori.run(transactions)
    apriori.print_results()
