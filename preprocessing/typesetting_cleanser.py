import re
from Wingene.factory import sentence_segmenter_factory
from Wingene.factory import typesetting_cleanser_factory
# from Wingene.factory.sentence_segmenter_factory import SentenceSegmenterFactory
# from Wingene.factory.typesetting_cleanser_factory import TypesettingCleanserFactory


class TypesettingCleanser(object):
    def __init__(self) -> None:
        self._SENTENCE_SEGMENTATION = sentence_segmenter_factory.get_SentenceSegmenterByBullet()
        self._TYPESETTING_CLEANSER = typesetting_cleanser_factory.get_TypesettingCleanserForNHIBasisReport()
        # self._SENTENCE_SEGMENTATION = SentenceSegmenterFactory.get_NHIReportSentenceSegmenter()
        # self._TYPESETTING_CLEANSER = TypesettingCleanserFactory.get_NHIReportTypesettingCleanser()

    def __call__(self, string: str) -> str:
        sents = (sent for sent in self._SENTENCE_SEGMENTATION.segment(self._TYPESETTING_CLEANSER.cleanse(string)) if sent)
        text = '\n'.join(sents)
        text = re.sub(r'- ', '', text)
        text = re.sub(r'bil(\.)? ', 'bilateral ', text)
        return text
