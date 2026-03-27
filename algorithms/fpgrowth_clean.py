from collections import defaultdict, Counter
import pandas as pd
from itertools import combinations


class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.link = None
        self.children = {}


class FPTree:
    def __init__(self, transactions, min_support, sort_index):
        self.min_support = min_support
        self.root = FPNode(None, 0, None)
        self.header_table = {}
        self.sort_index = sort_index
        
        for transaction in transactions:
            sorted_items = [item for item in sort_index if item in transaction]
            self._insert_transaction(sorted_items, self.root, 1)

    def _insert_transaction(self, transaction, node, count):
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
        if item in self.header_table:
            current = self.header_table[item]
            while current.link:
                current = current.link
            current.link = node
        else:
            self.header_table[item] = node


class FPGrowthAlgorithm:
    def __init__(self, min_support=0.1, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transactions = []
        self.num_transactions = 0
        self.frequent_itemsets = {}
        self.association_rules = []

    def load_transactions(self, transaction_list):
        self.transactions = [list(trans) for trans in transaction_list]
        self.num_transactions = len(self.transactions)

    def get_item_support(self):
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
        items = sorted(fp_tree.header_table.keys(),
                      key=lambda x: self._get_support(fp_tree.header_table[x]))

        for item in items:
            new_prefix = prefix + [item]
            support = self._get_support(fp_tree.header_table[item])
            
            itemset_key = frozenset(new_prefix)
            frequent_itemsets[itemset_key] = support

            conditional_transactions = self._get_conditional_transactions(
                fp_tree.header_table[item]
            )

            if conditional_transactions:
                min_count = self.min_support * self.num_transactions
                conditional_items = self._filter_items(
                    conditional_transactions, min_count
                )

                if conditional_items:
                    sort_index = sorted(
                        conditional_items.keys(),
                        key=lambda x: conditional_items[x]
                    )

                    conditional_tree = FPTree(
                        conditional_transactions,
                        self.min_support,
                        sort_index
                    )

                    self.fp_growth(
                        conditional_tree,
                        new_prefix,
                        frequent_itemsets
                    )

    def _get_support(self, node):
        support = 0
        while node:
            support += node.count
            node = node.link
        return support / self.num_transactions

    def _get_conditional_transactions(self, node):
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
        frequent_itemsets = {}

        frequent_items = self.get_item_support()
        if not frequent_items:
            print("No frequent itemsets found")
            return {}

        for item, count in frequent_items.items():
            support = count / self.num_transactions
            frequent_itemsets[frozenset([item])] = support

        sort_index = sorted(
            frequent_items.keys(),
            key=lambda x: frequent_items[x]
        )

        fp_tree = FPTree(self.transactions, self.min_support, sort_index)

        self.fp_growth(fp_tree, [], frequent_itemsets)

        self.frequent_itemsets = defaultdict(dict)
        for itemset, support in frequent_itemsets.items():
            size = len(itemset)
            self.frequent_itemsets[size][itemset] = support

        return dict(self.frequent_itemsets)

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

                            antecedent_support = self.frequent_itemsets[
                                antecedent_size
                            ][antecedent]
                            confidence = support / antecedent_support

                            if confidence >= self.min_confidence:
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
        self.load_transactions(transaction_list)
        self.find_frequent_itemsets()
        self.generate_rules()
        
        return self.frequent_itemsets, self.association_rules

    def print_results(self):
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
    transactions = [
        ['Bread', 'Milk', 'Beer', 'Diapers'],
        ['Bread', 'Diapers', 'Beer', 'Eggs'],
        ['Milk', 'Diapers', 'Beer', 'Cola'],
    ]

    fpg = FPGrowthAlgorithm(min_support=0.1, min_confidence=0.6)
    fpg.run(transactions)
    fpg.print_results()
