"""
FP-Tree (Frequent Pattern Tree) Algorithm Implementation
Association Rule Mining - FP-Growth Algorithm

Algorithm:
1. Build FP-Tree from transaction database
2. Mine frequent patterns directly from FP-Tree without candidate generation
3. Generate association rules from frequent patterns
4. More efficient than Apriori as it avoids candidate generation
"""

from collections import defaultdict, Counter
import pandas as pd


class FPNode:
    """Represents a node in the FP-Tree"""
    
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.link = None  # Link to next node with same item
        self.children = {}


class FPTree:
    """Represents the FP-Tree structure"""
    
    def __init__(self, transactions, min_support, sort_index):
        self.min_support = min_support
        self.root = FPNode(None, 0, None)
        self.header_table = {}
        self.sort_index = sort_index
        
        # Build the tree
        for transaction in transactions:
            sorted_items = [item for item in sort_index if item in transaction]
            self._insert_transaction(sorted_items, self.root, 1)

    def _insert_transaction(self, transaction, node, count):
        """Insert a transaction into the FP-Tree"""
        if not transaction:
            return

        first = transaction[0]
        if first in node.children:
            child = node.children[first]
            child.count += count
        else:
            child = FPNode(first, count, node)
            node.children[first] = child
            self._update_header_table(first, child)

        remaining = transaction[1:]
        if remaining:
            self._insert_transaction(remaining, child, count)

    def _update_header_table(self, item, node):
        """Update the header table"""
        if item in self.header_table:
            current = self.header_table[item]
            while current.link:
                current = current.link
            current.link = node
        else:
            self.header_table[item] = node


class FPGrowthAlgorithm:
    def __init__(self, min_support=0.1, min_confidence=0.6):
        """
        Initialize FP-Growth algorithm
        
        Args:
            min_support (float): Minimum support threshold (0-1)
            min_confidence (float): Minimum confidence threshold (0-1)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transactions = []
        self.num_transactions = 0
        self.frequent_itemsets = {}
        self.association_rules = []

    def load_transactions(self, transaction_list):
        """Load transactions"""
        self.transactions = [list(trans) for trans in transaction_list]
        self.num_transactions = len(self.transactions)

    def get_item_support(self):
        """Calculate support for each individual item"""
        item_count = defaultdict(int)
        for transaction in self.transactions:
            for item in transaction:
                item_count[item] += 1

        min_count = self.min_support * self.num_transactions
        frequent_items = {
            item: count
            for item, count in item_count.items()
            if count >= min_count
        }

        return frequent_items

    def fp_growth(self, fp_tree, prefix, frequent_itemsets):
        """
        Mine frequent patterns from FP-Tree
        
        Args:
            fp_tree (FPTree): The FP-Tree structure
            prefix (list): Current prefix
            frequent_itemsets (dict): Container for frequent itemsets
        """
        # Get items sorted by support (ascending)
        items = sorted(fp_tree.header_table.keys(),
                      key=lambda x: self._get_support(fp_tree.header_table[x]))

        for item in items:
            new_prefix = prefix + [item]
            support = self._get_support(fp_tree.header_table[item])
            
            itemset_key = frozenset(new_prefix)
            frequent_itemsets[itemset_key] = support

            # Build conditional FP-Tree
            conditional_transactions = self._get_conditional_transactions(
                fp_tree.header_table[item]
            )

            if conditional_transactions:
                # Calculate support for items in conditional transactions
                min_count = self.min_support * self.num_transactions
                conditional_items = self._filter_items(
                    conditional_transactions, min_count
                )

                if conditional_items:
                    # Sort by support
                    sort_index = sorted(
                        conditional_items.keys(),
                        key=lambda x: conditional_items[x]
                    )

                    # Build conditional FP-Tree
                    conditional_tree = FPTree(
                        conditional_transactions,
                        self.min_support,
                        sort_index
                    )

                    # Recursive mining
                    self.fp_growth(
                        conditional_tree,
                        new_prefix,
                        frequent_itemsets
                    )

    def _get_support(self, node):
        """Get total support for a node"""
        support = 0
        while node:
            support += node.count
            node = node.link
        return support / self.num_transactions

    def _get_conditional_transactions(self, node):
        """Get conditional transactions from leaf node"""
        transactions = []
        while node:
            transaction = []
            parent = node.parent
            while parent.item is not None:
                transaction.append(parent.item)
                parent = parent.parent
            
            if transaction:
                for _ in range(node.count):
                    transactions.append(transaction)
            
            node = node.link
        
        return transactions

    def _filter_items(self, transactions, min_count):
        """Filter items by minimum count"""
        item_count = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_count[item] += 1

        return {
            item: count
            for item, count in item_count.items()
            if count >= min_count
        }

    def find_frequent_itemsets(self):
        """Find all frequent itemsets using FP-Growth"""
        frequent_itemsets = {}

        # Get initial frequent items
        frequent_items = self.get_item_support()
        if not frequent_items:
            print("No frequent itemsets found")
            return {}

        # Add 1-itemsets
        for item, count in frequent_items.items():
            support = count / self.num_transactions
            frequent_itemsets[frozenset([item])] = support

        # Sort items by support (ascending)
        sort_index = sorted(
            frequent_items.keys(),
            key=lambda x: frequent_items[x]
        )

        # Build initial FP-Tree
        fp_tree = FPTree(self.transactions, self.min_support, sort_index)

        # Mine patterns
        self.fp_growth(fp_tree, [], frequent_itemsets)

        # Organize by itemset size
        self.frequent_itemsets = defaultdict(dict)
        for itemset, support in frequent_itemsets.items():
            size = len(itemset)
            self.frequent_itemsets[size][itemset] = support

        return dict(self.frequent_itemsets)

    def generate_rules(self):
        """Generate association rules from frequent itemsets"""
        from itertools import combinations

        self.association_rules = []

        for k in self.frequent_itemsets:
            if k >= 2:
                for itemset, support in self.frequent_itemsets[k].items():
                    itemset_list = list(itemset)

                    for antecedent_size in range(1, k):
                        for antecedent in combinations(itemset_list, antecedent_size):
                            antecedent = frozenset(antecedent)
                            consequent = itemset - antecedent

                            # Calculate confidence
                            antecedent_support = self.frequent_itemsets[
                                antecedent_size
                            ][antecedent]
                            confidence = support / antecedent_support

                            if confidence >= self.min_confidence:
                                # Calculate lift
                                consequent_itemset = frozenset(consequent)
                                consequent_size = len(consequent)
                                consequent_support = self.frequent_itemsets[
                                    consequent_size
                                ][consequent_itemset]
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
        """Run the complete FP-Growth algorithm"""
        self.load_transactions(transaction_list)
        self.find_frequent_itemsets()
        self.generate_rules()
        
        return self.frequent_itemsets, self.association_rules

    def print_results(self):
        """Print results in formatted way"""
        print("=" * 80)
        print("FP-GROWTH ALGORITHM RESULTS")
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
    # Example usage
    transactions = [
        ['Bread', 'Milk', 'Beer', 'Diapers'],
        ['Bread', 'Diapers', 'Beer', 'Eggs'],
        ['Milk', 'Diapers', 'Beer', 'Cola'],
    ]

    fpg = FPGrowthAlgorithm(min_support=0.1, min_confidence=0.6)
    fpg.run(transactions)
    fpg.print_results()
