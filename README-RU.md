
Проект предназначен для распознавания иврит-текста, имитация рукописных шрифтов. 
Образцы шрифтов:
fonts/guttman-yad-brush.ttf
fonts/Gveret-Levin-Alef-Alef-Alef.jpg

Распознаёт не идеально, особенно ошибается с пробелами, но это лучше, чем ничего. 

## Быстрый старт

### Скачать файлы моделей 
И поместить из в каталог models в проекте (на одном уровне с fonts, hwocr и т. д.)
8 файлов:
https://drive.google.com/file/d/1AE0zWAAsLuDegk74EpGNIEX9zHilnbpW/view?usp=sharing
https://drive.google.com/file/d/1sIPL8GAp-XVeLjfgd-HlBV_Bhllj9Hv8/view?usp=sharing
https://drive.google.com/file/d/1I2O82lsYI0GRep-lBsQKg06k38nKKXfW/view?usp=sharing
https://drive.google.com/file/d/1PdAZGHZEsfAbEmA1_Ov3Za-a8mRAyBUP/view?usp=sharing
https://drive.google.com/file/d/1CFFJ_er9UPn4E_yw8aQgaDMinrD09LuF/view?usp=sharing
https://drive.google.com/file/d/1GCqJ0kPo2uKkFEsj-agEZqzF9W5qEUdr/view?usp=sharing
https://drive.google.com/file/d/1fCP4sr8q75QHcAjTEdQjcQx31n_6w4-M/view?usp=sharing
https://drive.google.com/file/d/1pdvnE92FRHm1e9xXfwLyo8LR5MRTP9O1/view?usp=sharing


# 7. распознать свой файл
python3 -m hwocr.cli recognize --image page.png --out-image runs/annotated.png

# 8. веб‑интерфейс: drag&drop картинки + редактор рамок + демо (см. «Как запустить сервер»)
python3 -m hwocr.cli serve --port 8009
