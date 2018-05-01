import codecs
import os
#import pygame
import os
from PIL import Image, ImageFont, ImageDraw

PATH = '/Users/yao/Google Drive/projects_ml/GAN_Theories/Datas/fonts'
def generate_charater():
    counter = 0
    start, end = (0x4E00, 0x9FA5)  #汉字编码的范围
    with codecs.open(os.path.join(PATH, "chinese.txt"), "wb", encoding="utf-8") as f:
        for codepoint in range(int(start),int(end)):
            f.write(chr(codepoint))  #写出汉字
            counter += 1
    print(counter)

def generate_figure(font_file):
    font = font_file.split('.')[0]
    chinese_dir = os.path.join(PATH, font)
    if not os.path.exists(chinese_dir):
        os.mkdir(chinese_dir)

    #pygame.init()
    start,end = (0x4E00, 0x9FA5) # 汉字编码范围
    for codepoint in range(int(start), int(end)):
        word = chr(codepoint)
        # font = pygame.font.Font(os.path.join(PATH, font_file), 64)
        # rtext = font.render(word, True, (0, 0, 0), (255, 255, 255))
        # pygame.image.save(rtext, os.path.join(chinese_dir, word + ".jpg"))
        im = Image.new("RGB", (64, 64), (255, 255, 255))
        dr = ImageDraw.Draw(im)
        font = ImageFont.truetype(os.path.join(PATH, font_file), 64)
        dr.text((0, 0), word, font=font, fill="#000000")
        im.save(os.path.join(chinese_dir, word + ".jpg"))
        size = os.path.getsize(os.path.join(chinese_dir, word + ".jpg"))
        if size < 692:
            os.remove(os.path.join(chinese_dir, word + ".jpg"))


if __name__=='__main__':
    #generate_charater()
    generate_figure('wt071.ttf')