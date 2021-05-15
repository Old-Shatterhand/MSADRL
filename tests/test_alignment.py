from unittest import TestCase

from utils.alignment import align_seq_seq, align_prof_seq, align_progressive
from utils.constants import LEFT, DIAG, UP
from utils.profile import Profile


class AlignmentTest(TestCase):

    def test_align_seq_seq_1(self):
        seq_a = "gaac"
        seq_b = "caagac"
        profile, _ = align_seq_seq(seq_a, seq_b, True, [LEFT, DIAG, UP])
        self.assertEqual(["gaa--c", "caagac"], profile.get_sequences())

    def test_align_seq_seq_2(self):
        seq_a = "caagac"
        seq_b = "gaac"
        profile, _ = align_seq_seq(seq_a, seq_b, True, [LEFT, DIAG, UP])
        self.assertEqual(["caagac", "-ga-ac"], profile.get_sequences())

    def test_align_prof_seq(self):
        profile = Profile(["gc-gc-cc", "gccgcgcc"])
        seq = "gcgccc"
        profile = align_prof_seq(seq, profile, [LEFT, DIAG, UP])
        self.assertEqual(["gc-gc-cc", "gccgcgcc", "gc-gc-cc"], profile.get_sequences())

    def test_align_progressive_1(self):
        seqs = ["ctattg", "ctaccg", "ctatgt"]
        profile = align_progressive([2, 0, 1], seqs)
        self.assertEqual(["ctatgt-", "ctat-tg", "ctac-cg"], profile.get_sequences())

    def test_align_progressive_2(self):
        seqs = ["ctattg", "ctaccg", "ctatgt"]
        profile = align_progressive([1, 2, 0], seqs)
        self.assertEqual(["ctaccg-", "cta-tgt", "ctattg-"], profile.get_sequences())
