"""Render final assignment Markdown with explicit hierarchy and formula images."""
import argparse
import html
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Table, TableStyle,
    Image, Spacer, PageBreak, CondPageBreak, KeepTogether)

def inline(value):
    value = html.escape(value)
    value = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', value)
    value = re.sub(r'`([^`]+)`', r'<font name="Code">\1</font>', value)
    value = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', value)
    return value.replace('\n', '<br/>')

def render(source):
    zh = source.stem.endswith('_ZH')
    workshop = '3' if 'Workshop3' in source.stem else '4'
    bodyfont = 'Chinese' if zh else 'Arial'
    boldfont = 'Chinese' if zh else 'ArialBold'
    body = ParagraphStyle('Body', fontName=bodyfont, fontSize=10.3,
        leading=15.0 if zh else 14.3, spaceAfter=7, allowWidows=0, allowOrphans=0,
        wordWrap='CJK' if zh else None)
    title = ParagraphStyle('Title', parent=body, fontName=boldfont, fontSize=19,
        leading=24, spaceAfter=12, textColor=colors.HexColor('#18334a'))
    h2 = ParagraphStyle('Question', parent=body, fontName=boldfont, fontSize=13,
        leading=18, spaceBefore=8, spaceAfter=8, keepWithNext=True,
        textColor=colors.HexColor('#18334a'))
    h3 = ParagraphStyle('Section', parent=body, fontName=boldfont, fontSize=11,
        leading=15, spaceBefore=6, spaceAfter=5, keepWithNext=True)
    question = ParagraphStyle('Prompt', parent=body, fontName=boldfont,
        keepWithNext=True, spaceBefore=4, spaceAfter=4)
    caption = ParagraphStyle('Caption', parent=body, fontSize=9,
        leading=12, textColor=colors.HexColor('#405163'), spaceAfter=8)
    cell = ParagraphStyle('Cell', parent=body, fontSize=8.7,
        leading=12, spaceAfter=0)
    head = ParagraphStyle('Head', parent=cell, fontName=boldfont, textColor=colors.white)
    width = 176 * mm
    flow = []
    text = source.read_text(encoding='utf-8')
    assert not re.search(r'\$[^\n]+\$|\\(?:frac|gamma|theta|sum)\b', text), 'Unrendered formula source'
    for block in re.split(r'\n\s*\n', text.strip()):
        block = block.strip()
        if block.startswith('# '):
            flow.append(Paragraph(inline(block[2:]), title))
        elif block.startswith('## '):
            label = block[3:]
            if re.match(r'Task 2\s*-\s*Question', label):
                flow.append(PageBreak())
            elif re.match(r'Task 1\s*-\s*Question 2', label):
                flow.append(CondPageBreak(150 * mm))
            else:
                flow.append(CondPageBreak(30 * mm))
            flow.append(Paragraph(inline(label), h2))
        elif block.startswith('### '):
            flow.append(Paragraph(inline(block[4:]), h3))
        elif block.startswith('#### '):
            flow.append(Paragraph(inline(block[5:]), h3))
        elif block.startswith('!['):
            match = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', block)
            assert match, block
            label, relative = match.groups()
            path = (source.parent / relative).resolve()
            assert path.is_relative_to(source.parent.resolve()) and path.exists(), path
            picture = Image(str(path))
            formula = 'formula' in path.stem
            if formula:
                scale = min(width / picture.imageWidth, 72 / 600)
            else:
                scale = min(width / picture.imageWidth, 130 * mm / picture.imageHeight)
            picture.drawWidth = picture.imageWidth * scale
            picture.drawHeight = picture.imageHeight * scale
            picture.hAlign = 'CENTER'
            if formula:
                flow.extend([picture, Spacer(1, 7)])
            else:
                flow.append(KeepTogether([picture, Spacer(1, 5), Paragraph(inline(label), caption)]))
        elif block.startswith('|'):
            rows = [[v.strip() for v in line.strip().strip('|').split('|')]
                    for line in block.splitlines() if line.strip().startswith('|')]
            rows = [row for row in rows if not all(re.fullmatch(r'[-:]+', v) for v in row)]
            n = len(rows[0])
            assert all(len(row) == n for row in rows), block
            fractions = {2:[.24,.76], 3:[.34,.33,.33], 4:[.28,.24,.24,.24],
                         5:[.24,.19,.19,.19,.19], 6:[.22,.16,.155,.155,.155,.155]}.get(n,[1/n]*n)
            content = [[Paragraph(inline(v), head if i == 0 else cell) for v in row] for i,row in enumerate(rows)]
            table = Table(content, colWidths=[width*f for f in fractions], repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#234a65')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eff4f7'),colors.white]),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),
                ('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),5),
                ('RIGHTPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#b8c7d1'))]))
            flow.extend([KeepTogether([table]) if len(rows)<=8 else table, Spacer(1,8)])
        elif block.startswith('- '):
            for line in block.splitlines():
                flow.append(Paragraph(inline(line),body))
        else:
            flow.append(Paragraph(inline(block), question if re.fullmatch(r'\*\*.+\*\*', block, re.S) else body))
    def footer(canvas, document):
        canvas.setFont('Arial',8)
        canvas.setFillColor(colors.HexColor('#566774'))
        canvas.drawString(17*mm,10*mm,f'ELEN90098 | Workshop {workshop} | Final Report')
        canvas.drawRightString(193*mm,10*mm,str(document.page))
    destination = source.with_suffix('.pdf')
    SimpleDocTemplate(str(destination),pagesize=(210*mm,297*mm),leftMargin=17*mm,
        rightMargin=17*mm,topMargin=15*mm,bottomMargin=17*mm,
        title=source.stem,author='Zhengnan Cao; Yuchen Yang').build(flow,onFirstPage=footer,onLaterPages=footer)
    print(destination)

if __name__ == '__main__':
    fontdir=Path('C:/Windows/Fonts')
    for name, filename in [('Arial','arial.ttf'),('ArialBold','arialbd.ttf'),('Chinese','simsun.ttc'),('Code','consola.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fontdir/filename)))
    pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialBold',italic='Arial',boldItalic='ArialBold')
    pdfmetrics.registerFontFamily('Chinese',normal='Chinese',bold='Chinese',italic='Chinese',boldItalic='Chinese')
    parser=argparse.ArgumentParser()
    parser.add_argument('sources',nargs='+',type=Path)
    for source in parser.parse_args().sources:
        render(source.resolve())
