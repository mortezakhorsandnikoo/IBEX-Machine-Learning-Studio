"""Create the printable guide from the same catalog used by the app."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image, PageBreak, KeepTogether, CondPageBreak
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/pdf/IBEX_Machine_Learning_Studio_User_Guide.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
TOPICS = json.loads((ROOT/'Help/guide.json').read_text(encoding='utf-8'))
INDEX = {t['Id']:t for t in TOPICS}
GREEN, NAVY, GRAY = map(colors.HexColor, ['#087F73','#172B45','#59697B'])
for name,file in [('Calibri','calibri.ttf'),('Calibri-Bold','calibrib.ttf'),('Calibri-Italic','calibrii.ttf')]:
    pdfmetrics.registerFont(TTFont(name, 'C:/Windows/Fonts/'+file))
pdfmetrics.registerFontFamily('Calibri',normal='Calibri',bold='Calibri-Bold',italic='Calibri-Italic',boldItalic='Calibri-Bold')
styles = {
 'body': ParagraphStyle('body',fontName='Calibri',fontSize=11,leading=15,textColor=NAVY,spaceAfter=8),
 'chapter': ParagraphStyle('chapter',fontName='Calibri-Bold',fontSize=25,leading=29,textColor=NAVY,spaceAfter=18,keepWithNext=True),
 'topic': ParagraphStyle('topic',fontName='Calibri-Bold',fontSize=18,leading=22,textColor=NAVY,spaceBefore=14,spaceAfter=9,keepWithNext=True),
 'heading': ParagraphStyle('heading',fontName='Calibri-Bold',fontSize=13,leading=17,textColor=GREEN,spaceBefore=8,spaceAfter=6,keepWithNext=True),
 'summary': ParagraphStyle('summary',fontName='Calibri',fontSize=11,leading=15,textColor=GRAY,spaceAfter=12,keepWithNext=True),
 'caption': ParagraphStyle('caption',fontName='Calibri',fontSize=9,leading=12,textColor=GRAY,spaceAfter=12),
 'note': ParagraphStyle('note',fontName='Calibri',fontSize=10.5,leading=14,textColor=NAVY,backColor=colors.HexColor('#EDF6F3'),borderPadding=9,spaceBefore=8,spaceAfter=14),
 'item': ParagraphStyle('item',fontName='Calibri',fontSize=11,leading=15,textColor=NAVY,leftIndent=18,firstLineIndent=0,bulletIndent=0,spaceAfter=7),
 'kicker': ParagraphStyle('kicker',fontName='Calibri-Bold',fontSize=10,leading=14,textColor=GREEN,spaceAfter=12),
 'title': ParagraphStyle('title',fontName='Calibri-Bold',fontSize=36,leading=40,textColor=NAVY,spaceAfter=14),
}
for style in styles.values():
    style.allowWidows = 0
    style.allowOrphans = 0
def clean(text):
    return text.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-').replace('\u2212','-').replace('’',"'").replace('“','"').replace('”','"').replace('•','|')
def para(text,style='body',**kwargs): return Paragraph(escape(clean(text)),styles[style],**kwargs)

class Guide(BaseDocTemplate):
    def __init__(self):
        super().__init__(str(OUT),pagesize=(612,792),leftMargin=52,rightMargin=52,topMargin=55,bottomMargin=50,
                         title='IBEX Machine Learning Studio - User Guide',author='Morteza Khorsand Nikoo',
                         subject='Complete illustrated guide to datasets, preprocessing and all algorithms')
        self.addPageTemplates(PageTemplate('guide',frames=Frame(52,50,508,687,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0),onPage=self.decorate))
    def decorate(self,canv,doc):
        if doc.page == 1: return
        canv.saveState(); canv.setFillColor(GREEN); canv.setFont('Calibri-Bold',9)
        canv.drawString(52,764,'IBEX  /  MACHINE LEARNING STUDIO')
        canv.setStrokeColor(colors.HexColor('#D9E5E2'));canv.line(52,754,560,754)
        canv.setFillColor(GRAY);canv.setFont('Calibri',9)
        canv.drawString(52,28,'USER GUIDE  |  September 2026')
        canv.drawRightString(560,28,str(doc.page));canv.restoreState()
    def afterFlowable(self,f):
        if hasattr(f,'bookmark'):
            self.canv.bookmarkPage(f.bookmark)
            self.canv.addOutlineEntry(f.getPlainText(),f.bookmark,f.level,closed=False)
            self.notify('TOCEntry',(f.level,f.getPlainText(),self.page,f.bookmark))

def mark(text,style,key,level):
    q=para(text,style);q.bookmark=key;q.level=level;return q
def figure(path,caption,maxheight=320):
    w,h=ImageReader(str(path)).getSize(); scale=min(508/w,maxheight/h)
    return KeepTogether([Spacer(1,6),Image(str(path),w*scale,h*scale,hAlign='CENTER'),Spacer(1,6),para(caption,'caption')])

story=[Spacer(1,32)]
story.append(figure(ROOT/'Assets/ibex.png','',maxheight=125))
story += [Spacer(1,25),para('IBEX','title'),para('Machine Learning Studio','chapter'),
          para('Complete user guide','title'),para('Explore. Train. Understand.','summary'),Spacer(1,18),
          para('From your first CSV to preprocessing, model evaluation and prediction. Illustrated instructions for all 24 algorithms.'),
          para('50 searchable topics • 12 illustrations','kicker'),Spacer(1,25),
          para('Designed by Morteza Khorsand Nikoo','heading'),
          Paragraph('<link href="mailto:morteza.khorsandnikoo@uibk.ac.at" color="#087F73">morteza.khorsandnikoo@uibk.ac.at</link>',styles['body']),
          para('September 2026','caption'),PageBreak(),para('Contents','chapter'),
          para('Click a topic or use the PDF bookmarks. Use Ctrl+F in your PDF reader to search the full text. In the app, open Help and user guide or press F1.'),Spacer(1,10)]
toc=TableOfContents()
toc.levelStyles=[ParagraphStyle('toc0',fontName='Calibri-Bold',fontSize=11,leading=15,textColor=GREEN,spaceBefore=9),
                 ParagraphStyle('toc1',fontName='Calibri',fontSize=10,leading=13,textColor=NAVY,leftIndent=14,firstLineIndent=0,spaceBefore=3)]
story.append(toc)
categories=['Getting started','Dataset','Prepare data','Regression','Regression algorithms','Classification','Classification algorithms','Clustering','Clustering algorithms','Evaluation','Prediction and saving','Insights','Reference']
figure_count=0
for chapter,category in enumerate(categories,1):
    story += [PageBreak() if chapter == 1 else CondPageBreak(220),Spacer(1,16),para(f'{chapter:02d}  /  USER GUIDE','kicker'),mark(category,'chapter','category-'+str(chapter),0)]
    for t in [x for x in TOPICS if x['Category']==category]:
        story += [CondPageBreak(135),mark(t['Title'],'topic',t['Id'],1),para(t['Summary'],'summary')]
        for b in t['Blocks']:
            kind=b['Kind']
            if kind=='image':
                figure_count+=1
                story.append(figure(ROOT/'Help/Images'/b['Image'],f'Figure {figure_count}. '+b['Caption'],430 if b['Image']=='correlation.png' else 320))
            elif kind in ('steps','bullets'):
                for n,item in enumerate(b['Items'],1):
                    story.append(para(item,'item',bulletText=str(n)+'.' if kind=='steps' else '-'))
            else: story.append(para(b.get('Text',''),kind if kind in ('heading','note') else 'body'))
        if t['Related']:
            links=' &nbsp; | &nbsp; '.join('<link href="#'+id+'" color="#087F73">'+escape(clean(INDEX[id]['Title']))+'</link>' for id in t['Related'])
            previous = story.pop()
            story.append(KeepTogether([previous,Paragraph('<b>Related:</b> '+links,styles['caption'])]))
        story.append(Spacer(1,8))
Guide().multiBuild(story)
print(OUT)
