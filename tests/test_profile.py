from unittest import TestCase

from utils.profile import Profile
from utils.utils import GAP, score_dna_match


class ProfileTest(TestCase):
    def test_add_gap_1(self):
        profile = Profile(["agct", "ag-t"])
        profile.add_gap(0)
        self.assertEqual(GAP, profile.seqs[0][0], "Gap in seq1 not inserted")
        self.assertEqual(GAP, profile.seqs[1][0], "Gap in seq1 not inserted")
        self.assertEqual(5, len(profile.sp_scores), "SP-Scores has not correct length")
        self.assertEqual(score_dna_match, profile.sp_scores[2], "SP-Score not recomputed")
        self.assertEqual(5, len(profile), "Profile has not correct size")

    def test_add_gap_2(self):
        profile = Profile(["agct", "ag-t"])
        profile.add_gap(2)
        self.assertEqual(GAP, profile.seqs[0][2], "Gap in seq1 not inserted")
        self.assertEqual(GAP, profile.seqs[1][2], "Gap in seq2 not inserted")
        self.assertEqual(5, len(profile.sp_scores), "SP-Scores has not correct length")
        self.assertEqual(score_dna_match, profile.sp_scores[4], "SP-Score is not recomputed")
        self.assertEqual(5, len(profile), "Profile has not correct size")

    def test_add_gap_3(self):
        profile = Profile(["agct", "ag-t"])
        profile.add_gap(4)
        self.assertEqual(GAP, profile.seqs[0][4], "Gap in seq1 not inserted")
        self.assertEqual(GAP, profile.seqs[1][4], "Gap in seq2 not inserted")
        self.assertEqual(5, len(profile.sp_scores), "SP-Scores has not correct length")
        self.assertEqual(score_dna_match, profile.sp_scores[3], "SP-Score not recomputed")
        self.assertEqual(5, len(profile), "Profile has not correct size")
