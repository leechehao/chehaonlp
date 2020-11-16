import re


class QuaDetect(object):
    def __init__(self):
        L = "".join((r"\[", chr(40), chr(12304), chr(65308), chr(65288)))
        R = "".join((r"\]", chr(41), chr(12305), chr(65310), chr(65289)))
        V = "".join((chr(45), chr(42), chr(43), chr(79), chr(118)))
        B = "".join((chr(9632), chr(9633), chr(9679), chr(9675)))

        r_1 = "".join(("[", L, "][ ]*[", V, "]?[ ]*[", R, "]"))
        r_2 = "".join(("[", B, "]"))

        self.p_1 = re.compile(r_1)
        self.p_2 = re.compile(r_2)

        self.cut = re.compile("[ \t\v]+")

    def check_table(self, string):
        for p in (self.p_1, self.p_2):
            if p.search(string) is not None:
                return True
        return False

    def check_crash(self, string):
        words = [substring for substring in self.cut.split(string) if substring]

        len_words = len(words)

        if len_words <= 0:
            return False

        avg_len_for_words = sum((len(word) for word in words)) / len_words

        if len_words <= 5 and avg_len_for_words >= 20:
            return True
        if len_words > 5 and avg_len_for_words > 10:
            return True
        return False
