import easyocr
import pprint


class EasyOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en','ja'], gpu=True)
        self.languages = ['en','ja']

    def set_language(self, languages):
        self.languages = languages
        self.reader = easyocr.Reader(languages)

    def predict(self, image):
        results = list(self.reader.readtext(image, width_ths=0.7))
        new_res = []
        for res in results:
            box = res[0]
            text = res[1]
            conf = res[2]
            # covnert box points to int
            for idx, point in enumerate(box):
                box[idx] = [int(p) for p in point]
            new_res.append({'box': box, 'text': text, 'conf': float(conf)})
        return new_res

    def get_languages(self):
        return self.languages


def test():
    easy_ocr = EasyOCR()
    easy_ocr.predict('../tests/sample.jpg')


if __name__ == '__main__':
    test()

