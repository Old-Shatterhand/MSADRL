import random
from unittest import TestCase

import numpy as np

from utils.hash_align_table import HashAlignTable
from utils.profile import Profile, StoreProfile
from utils.utils import linearize_complete_state, hash_state_fast, get_sequences


class HashAlignTableTest(TestCase):
    def test_set(self, profile, permutation, sequences):
        table = HashAlignTable(sequences)
        hash_value = hash_state_fast(permutation, len(sequences))

        table.set(permutation, profile)

        self.assertEqual(profile, table.table[hash_value].to_profile(sequences))

    def test_get(self, profile, permutation, sequences):
        table = HashAlignTable(sequences)
        hash_value = hash_state_fast(permutation, len(sequences))

        table.table[hash_value] = StoreProfile(profile, permutation)

        self.assertEqual(profile, table.get(permutation)[0])

    def test_fuzzer(self, count=1):
        '''
        for _ in range(count):
            num_seqs = np.random.randint(1, 5)
            seqs_count = np.random.randint(num_seqs, 10)
            profile = Profile(get_sequences(length=10, count=num_seqs, different=True))
            permutation = random.sample(range(seqs_count), num_seqs)
            self.test_get(profile, permutation, )
            self.test_set(profile, permutation)
        '''
        sequences = ["ACGT", "AGT", "ACT"]
        permutation = [2, 0, 1]
        profile = Profile(["AC-T", "ACGT", "A-GT"])
        self.test_set(profile, permutation, sequences)
        self.test_get(profile, permutation, sequences)
