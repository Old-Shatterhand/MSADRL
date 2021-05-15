from unittest import TestCase

from utils.utils import *


class UtilsTest(TestCase):
    def test_linearize_state_1(self):
        state = [2, 0, 1]
        result = linearize_state(state, 3)
        self.assertEqual([0, 0, 1, 1, 0, 0], result)

    def test_linearize_state_2(self):
        state = [2, 0]
        result = linearize_state(state, 4)
        self.assertEqual([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], result)

    def test_linearize_complete_state_1(self):
        state = [2, 0, 1]
        result = linearize_complete_state(state, 3)
        self.assertEqual([0, 0, 1, 1, 0, 0, 0, 1, 0], result)

    def test_linearize_complete_state_2(self):
        state = [2, 0]
        result = linearize_complete_state(state, 4)
        self.assertEqual([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], result)

    def test_hash_state_1(self):
        state = [2, 0, 1]
        result = hash_state(linearize_state(state, 3), 3)
        self.assertEqual(int('20', 3), result)

    def test_hash_state_2(self):
        state = [2, 0]
        result = hash_state(linearize_state(state, 4), 4)
        self.assertEqual(int('200', 4), result)
