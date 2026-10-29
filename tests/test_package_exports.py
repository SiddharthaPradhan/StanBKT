import stanbkt
from stanbkt import models, utils


class TestTopLevelExports:
    def test_all_names_importable(self):
        for name in stanbkt.__all__:
            assert hasattr(stanbkt, name), name

    def test_top_level_matches_submodules(self):
        assert stanbkt.MultiBKT is models.MultiBKT
        assert stanbkt.MultiPriors is models.MultiPriors
        assert stanbkt.InitKnowledgeStrategy is models.InitKnowledgeStrategy
        assert stanbkt.sim_grouped_BKT is utils.sim_grouped_BKT
