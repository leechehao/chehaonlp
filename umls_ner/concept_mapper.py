import re
from typing import List
from typing import Tuple
from typing import Sequence
from typing import Generator
from Wingene.ontology.UMLS import variable
from Wingene.factory import entity_classifier_factory
from Wingene.factory import entity_recognizer_factory
from Wingene.factory import substring_enumerator_factory
# from Wingene.enumeration.enum_of_UMLS_ontology import UMLS_UI_Type
# from Wingene.factory.entity_classifier_factory import EntityClassifierFactory
# from Wingene.factory.entity_recognizer_factory import EntityRecognizerFactory
# from Wingene.data_type.anchor import Anchor
# from Wingene.data_type.regex import RegexChecker
# from Wingene.ontology.UMLS.static_variable import stop_word_Trie
# from Wingene.alogrithm.sequence.contiguous_subsequence import ContiguousSubsequence
# from Wingene.natural_language_processing.word_sliding.word_slider import WordSlider


class ConceptMapper(object):
    def __init__(self, TUIs: Tuple[str] = None, UI_type = None, min_count : int = None, max_count: int = None):
        UMSL_entity_classifier = entity_classifier_factory.get_UMLSEntityClassifier(TUIs=TUIs, UI_type=UI_type)
        self.UMLS_Recognizer = entity_recognizer_factory.get_StandardEntityRecognizer(
            substring_enumerator_factory.get_SubstringEnumeratorForSymptomExtraction(
                min_count,
                max_count,
                substring_enumerator_factory.get_UMLSSubstringEnumerator(min_count, max_count)
            ),
            UMSL_entity_classifier
        )

        # entity_classifier = EntityClassifierFactory.get_UMLSEntityClassifier(sTUIs=sTUIs, UI_type=UI_type)
        # self.recognizer = EntityRecognizerFactory.get_CommonEntityRecognizer(entity_classifier, WordFilter(min_count, max_count))

    def __call__(self, words: List[str]) -> Generator:
        for bouy in self.UMLS_Recognizer.recognize(words):
            yield(bouy.get_anchor().get_start_idx(), bouy.get_anchor().get_end_idx(), set(str(ent) for ent in bouy.get_cargo()))

        # for anchor in self.recognizer.recognize(words):
        #     yield(anchor.get_start_idx(), anchor.get_end_idx(), set(str(ent) for ent in anchor.get_objects()))

if __name__ == '__main__':

    UMLS_TUI = ConceptMapper(TUIs=("T023", "T029", "T030", "T046", "T047",
                                    "T082", "T191"),
                             UI_type=variable.UIType.TUI,
                             min_count=1,
                             max_count=4)

    UMLS_CUI = ConceptMapper(TUIs=('T184', 'T019', 'T052', 'T020', 'T080',
                                    'T047', 'T024', 'T017', 'T191', 'T190',
                                    'T023', 'T081', 'T169', 'T031', 'T033',
                                    'T167', 'T082', 'T078', 'T079', 'T042',
                                    'T121', 'T197'),
                             UI_type=variable.UIType.CUI,
                             min_count=1,
                             max_count=4)

    CUI_sets = [[
            "C0032225", "C0225968", "C1522720", "C0442121", "C0442045",
            "C0444498", "C0205150", "C1522619", "C0450219", "C0227843",
            "C0205052", "C4269139", "C0595836", "C0332469", "C0443268",
            "C0444471", "C1512948", "C0442143", "C0589496", "C1515009",
            "C0007570"
    ],
    [
            "C0522501", "C0444667", "C0205164", "C1542147", "C0205165"
    ],
    [
            "C2750120", "C0205400", "C3669021", "C1511606", "C1518633",
            "C4321392", "C0221198", "C2985135", "C1513916", "C0332513",
            "C3827002", "C4554646", "C1963113", "C0577559", "C3273930",
            "C0018800", "C4697773", "C0332461", "C2242558", "C0038002",
            "C4716264", "C1334928", "C1265608", "C0700124", "C0265512",
            "C0333348", "C0332562", "C0333482", "C0333159", "C0231556",
            "C0700140", "C2963144", "C0205307", "C0334129", "C0242362",
            "C0332511", "C0205266", "C0028259", "C0017525", "C0001863",
            "C0024204", "C0222032", "C0746922", "C0005889", "C0006736",
            "C0175566", "C0002940", "C0302142", "C0012817", "C0264358",
            "C0266631", "C0026727", "C0332523", "C0332468", "C2887948",
            "C0037157", "C0449579", "C0013604", "C0018802", "C1510420",
            "C0241311", "C2063664", "C0007137", "C1800706", "C0445177",
            "C0019270", "C0015734", "C0741611", "C0264511", "C0032790",
            "C0439688", "C3272301", "C2826609", "C0332514", "C3163633"
    ],
    [
            "C0205413", "C0205208", "C1709160", "C0205297", "C0175895",
            "C0205332", "C1881065", "C0442800", "C0205189", "C0205250",
            "C0392756", "C1512560", "C0449584", "C0205271", "C0205183",
            "C0036525", "C0392755", "C0332663", "C0443172", "C0225317",
            "C0178587", "C0205234", "C0442080", "C0599946", "C0205191",
            "C0205132", "C3897428", "C1704972", "C0439645", "C0332490",
            "C1948029", "C0205207", "C0439742", "C0205370", "C0027540",
            "C0750540", "C3714811", "C0444099", "C3854260", "C4527217",
            "C0205282", "C0522499", "C0205617", "C0205330", "C2945760",
            "C1609982", "C0332261", "C0439794", "C0205403", "C1709367",
            "C0205194", "C0439745", "C0439739", "C1282914", "C0205122",
            "C1514164", "C1979848", "C0205417", "C0205147", "C0221928",
            "C4554553", "C0026136", "C0019409"
    ],
    [
            "C2945599", "C0547040", "C0439064", "C0205388", "C0205392",
            "C1704332", "C4554554", "C0205314", "C0205081", "C0205447",
            "C0205448", "C0205449", "C0205450", "C0205451", "C0205452",
            "C0205453", "C0205454", "C0205455", "C0205456", "C0443295",
            "C2937276"
    ],
    [
        "C0700321"
    ],
    [
        "C0456389", "C0205219", "C0205100", "C0205284", "C0205402",
        "C0549177"
    ]
    ]
    CUI_cls = ["Location", "Spatial", "Symptom", "Modify", "Number", "Range", "General"]
    from collections import defaultdict

    words = 'Two Lung nodules in right middle lobe .'.split(' ')
    label_seq = ['Other']*len(words)

    tui_dict_level = {"T023": 2, "T029": 2, "T030": 2, "T082": 1, "T046": 0, "T047": 0, "T191": 0}
    tui_dict = defaultdict(lambda: 'Symptom')
    tui_dict[1] = 'Spatial'
    tui_dict[2] = 'Location'

    for start_idx, end_idx, tui_list in UMLS_TUI(words):
        label = tui_dict[max(tui_dict_level[tui] for tui in tui_list)]
        print(words[start_idx:end_idx], label)

    len_CUI_sets = len(CUI_sets)
    for start_idx, end_idx, cui_list in UMLS_CUI(words):
        for j in range(len_CUI_sets):
            if any(cui in CUI_sets[j] for cui in cui_list):
                print(words[start_idx:end_idx], cui_list, CUI_cls[j])


# class WordFilter(WordSlider):
#     def __init__(self, min_count: int = None, max_count: int = None) -> None:
#         self._STOP_WORDS = stop_word_Trie.get_all_words()
#         self._REGEX_CHECKER_FOR_NORMAL_CHAR = RegexChecker(
#             (
#                 re.compile(r"[0-9a-z]", re.IGNORECASE),
#             )
#         )
#         self._REGEX_CHECKER_FOR_NON_NORMAL_CHAR = RegexChecker(
#             (
#                 re.compile(r"[^a-z\d\s]", re.IGNORECASE),
#             )
#         )
#         self._REGEX_CHECKER_FOR_Num = RegexChecker(
#             (
#                 re.compile(r"^\d+$", re.IGNORECASE),
#             )
#         )        
#         super(WordFilter, self).__init__(min_count, max_count)

#     def enum(self, words: Sequence[str]) -> Generator[Anchor, None, None]:
#         for anchor in ContiguousSubsequence.enum_for_anchors(words, self.get_min_window_size(), self.get_max_window_size()):
#             has_noise_word = False
#             new_words = anchor.get_subsequence(words)
#             for idx in (0, -1):  # head and tail
#                 word = new_words[idx]
#                 if word in self._STOP_WORDS or (not self._REGEX_CHECKER_FOR_NORMAL_CHAR.check(word)):
#                     has_noise_word = True

#             text = ' '.join(new_words)
#             if self._REGEX_CHECKER_FOR_NON_NORMAL_CHAR.check(text):
#                 has_noise_word = True
#             elif any(self._REGEX_CHECKER_FOR_Num.check(word) for word in new_words):
#                 has_noise_word = True

#             if not has_noise_word:
#                 yield anchor
