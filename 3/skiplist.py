import collections.abc


class SkipList(collections.abc.MutableSet, collections.abc.Sequence):
    def __init__(self, data):
        self._data = list(data)

    def __getitem__(self, index):
        return self._data[index]

    def __contains__(self, item):
        return item in self._data

    def __iter__(self):
        for element in self._data:
            yield element

    def __or__(self, l2):
        return list(set(self._data) | set(l2))

    def __and__(self, l2):
        return list(set(self._data) & set(l2))

    def __len__(self):
        return len(self._data)

    def add(self):
        pass

    def discard(self):
        pass



import unittest
class TestSkipList(unittest.TestCase):
    def setUp(self):
        # Prime indices 2,7,12,17,22,27,32,...
        prime = [3,17,37,59,79,103,131,181,211,239,269,293,331,359,389]
        # Fibonnaci Prime
        fibprime = [3,4,5,7,11,13,17,23,29,43,47,83,131,137,359,431,433,449]

        self.union_prime_fibprime = list(set(prime) | set(fibprime))
        self.intersect_prime_fibprime = list(set(prime) & set(fibprime))

        # SkipList Setup
        self.l1 = SkipList(prime)
        self.l2 = SkipList(fibprime)

    def test_union(self):
        l3 = self.l1 | self.l2 
        self.assertSequenceEqual(l3, SkipList(self.union_prime_fibprime))


    def test_intersect(self):
        l3 = self.l1 & self.l2 
        self.assertSequenceEqual(l3, SkipList(self.intersect_prime_fibprime))

        

if __name__ == '__main__':
    unittest.main()
