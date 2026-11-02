from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT=Path(__file__).resolve().parents[2]
IMG=ROOT/'image'/'regression'/'linear'
OUT=ROOT/'docs'/'IBEX_User_Guide_Preview.docx'
doc=Document(); sec=doc.sections[0]
sec.page_width=Inches(8.5); sec.page_height=Inches(11)
sec.top_margin=Inches(.63); sec.bottom_margin=Inches(.62)
sec.left_margin=sec.right_margin=Inches(.72)
sec.header_distance=sec.footer_distance=Inches(.28)
sec.different_first_page_header_footer=True
W=7.06; GREEN='087F73'; NAVY='172B45'; GRAY='59697B'
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
    s=doc.styles[name]; s.font.name='Calibri'; s.font.color.rgb=RGBColor.from_string(NAVY)
    s.paragraph_format.space_after=Pt(6)
doc.styles['Normal'].font.size=Pt(11)
doc.styles['Normal'].paragraph_format.line_spacing=1.06
doc.styles['Title'].font.size=Pt(34); doc.styles['Title'].font.bold=True
doc.styles['Title'].font.color.rgb=RGBColor(0,0,0)
doc.styles['Subtitle'].font.size=Pt(20); doc.styles['Subtitle'].font.color.rgb=RGBColor(0,0,0)
doc.styles['Heading 1'].font.size=Pt(23)
doc.styles['Heading 1'].paragraph_format.space_after=Pt(10)
doc.styles['Heading 1'].paragraph_format.space_before=Pt(0)
doc.styles['Heading 2'].font.size=Pt(14); doc.styles['Heading 2'].font.color.rgb=RGBColor.from_string(GREEN)
doc.styles['Heading 2'].paragraph_format.space_before=Pt(8)
doc.styles['Caption'].font.size=Pt(9); doc.styles['Caption'].font.italic=False
doc.styles['Caption'].font.color.rgb=RGBColor.from_string(GRAY)
doc.core_properties.title='IBEX Machine Learning Studio User Guide'
doc.core_properties.author='Morteza Khorsand Nikoo'
doc.core_properties.subject='Getting started and your first regression model'

def p(text='',style=None,parent=doc,size=None,color=None):
    x=parent.add_paragraph(style=style)
    for i,part in enumerate(text.split('**')):
        r=x.add_run(part); r.bold=i%2==1
        if size: r.font.size=Pt(size)
        if color: r.font.color.rgb=RGBColor.from_string(color)
    return x

def kicker(text):
    x=p('**'+text.upper()+'**',size=9,color=GREEN); x.paragraph_format.space_after=Pt(9)
    return x

def page(n,title,intro):
    x=kicker(f'{n:02d}  Getting started'); x.paragraph_format.page_break_before=True
    p(title,'Heading 1'); p(intro)

def image(path,caption,width=W,parent=doc):
    x=parent.add_paragraph(); x.paragraph_format.space_after=Pt(3); x.paragraph_format.keep_with_next=True
    pic=x.add_run().add_picture(str(path),width=Inches(width)); pic._inline.docPr.set('descr',caption)
    p(caption,'Caption',parent=parent)

def shade(c,fill):
    el=OxmlElement('w:shd');el.set(qn('w:fill'),fill);c._tc.get_or_add_tcPr().append(el)

def table(headers,rows,widths,size=10):
    t=doc.add_table(rows=1,cols=len(headers));t.autofit=False
    for col,w in zip(t.columns,widths): col.width=Inches(w)
    for j,h in enumerate(headers):
        c=t.rows[0].cells[j];c.width=Inches(widths[j]);shade(c,GREEN)
        r=c.paragraphs[0].add_run(h);r.bold=True;r.font.size=Pt(size);r.font.color.rgb=RGBColor(255,255,255)
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
    for i,row in enumerate(rows):
        cells=t.add_row().cells
        for j,txt in enumerate(row):
            c=cells[j];c.width=Inches(widths[j])
            if i%2==0:shade(c,'F2F6F8')
            x=c.paragraphs[0];x.paragraph_format.space_before=Pt(3);x.paragraph_format.space_after=Pt(3);x.paragraph_format.line_spacing=1.02
            r=x.add_run(txt);r.font.size=Pt(size);r.bold=j==0
    for row in t.rows:
        row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
        for c in row.cells:
            mar=OxmlElement('w:tcMar')
            for edge in ['top','left','bottom','right']:
                e=OxmlElement('w:'+edge);e.set(qn('w:w'),'55' if edge in ['top','bottom'] else '75');e.set(qn('w:type'),'dxa');mar.append(e)
            c._tc.get_or_add_tcPr().append(mar)
    x=doc.add_paragraph();x.paragraph_format.space_after=Pt(0);x.paragraph_format.space_before=Pt(0);x.paragraph_format.line_spacing=Pt(3)

def note(title,text):
    t=doc.add_table(rows=1,cols=1);t.autofit=False;t.columns[0].width=Inches(W)
    c=t.cell(0,0);shade(c,'EDF6F3')
    x=c.paragraphs[0];x.paragraph_format.space_before=Pt(7);x.paragraph_format.space_after=Pt(4)
    r=x.add_run(title);r.bold=True;r.font.size=Pt(10);r.font.color.rgb=RGBColor.from_string(GREEN)
    x=p(text,parent=c,size=10);x.paragraph_format.space_after=Pt(8)
    for x in c.paragraphs:x.paragraph_format.left_indent=Pt(9);x.paragraph_format.right_indent=Pt(9)
    x=doc.add_paragraph();x.paragraph_format.line_spacing=Pt(3);x.paragraph_format.space_after=Pt(0)

def step(n,label,text):p(f'**{n}  {label}**  {text}')

h=sec.header.paragraphs[0];h.text='IBEX MACHINE LEARNING STUDIO     USER GUIDE'
h.runs[0].font.size=Pt(8);h.runs[0].font.color.rgb=RGBColor.from_string(GRAY)
for footer in [sec.footer,sec.first_page_footer]:
    x=footer.paragraphs[0];x.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    r=x.add_run('IBEX User Guide   |   ');r.font.size=Pt(8);r.font.color.rgb=RGBColor.from_string(GRAY)
    f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');x._p.append(f)

kicker('User guide')
x=doc.add_paragraph();x.paragraph_format.space_after=Pt(18)
x.add_run().add_picture(str(ROOT/'Assets'/'ibex.png'),width=Inches(1.65))
p('IBEX Machine\nLearning Studio','Title')
p('Getting started and your first model','Subtitle')
p('A practical guide to importing data, preparing features, training models and understanding predictions. Follow the on-screen control names in bold and use the illustrated checks to confirm each stage.',size=12)
p('What you will do','Heading 2')
p('Start from Home, import a CSV, inspect the dataset, choose appropriate preprocessing and train a linear regression model. You will also learn how to enter original feature values for prediction and save your results.')
table(['In this guide','Page'],[('Find your way around the studio','2'),('Import and inspect your dataset','3'),('Prepare features and apply PCA','4'),('Train and evaluate a regression model','5')],[6.35,.71],11)
note('The illustrated dataset','The supplied example contains 768 rows, eight input columns named X1 to X8 and one target named y1. These names are retained throughout the examples; use the meanings and units defined for your own dataset.')
p('**Designed by Morteza Khorsand Nikoo**')
p('morteza.khorsandnikoo@uibk.ac.at',size=10,color=GREEN)

page(1,'Find your way around the studio','Open the application to begin at Home. The three cards lead to Regression, Classification and Clustering. For a new analysis, first select **Dataset** in the left sidebar and import your CSV.')
image(ROOT/'Tests'/'verification'/'home.png','Figure 1  Home introduces the three modeling tasks. Use the sidebar to move between pages.',6.15)
table(['Page','What you do here'],[
('Dataset','Import a CSV and inspect rows, statistics, distributions and correlations.'),
('Prepare data','Describe, clean, encode, normalize and reduce the dataset with PCA.'),
('Regression','Predict a numeric outcome, such as y1 in this example.'),
('Classification','Predict a class from labeled examples.'),
('Clustering','Discover groups without choosing a target column.'),
('Compare models','Review recorded regression and classification runs.'),
('Visualize','Explore selected columns with charts and optional ascending X sorting.')],[1.45,5.61],10)
p('Working across pages','Heading 2')
p('You can move between pages without clearing the workspace. **Reset page** clears results on that page only. On Prepare data it clears the reports, not the prepared dataset used by other pages. Importing another dataset replaces the active data; save any models or results you need first.',size=10.5)

page(2,'Import and inspect your dataset','A feature is an input used by a model. A target is the value you want to predict. Here, X1 to X8 are features and y1 is the regression target.')
step(1,'Prepare the CSV','Use a header row, comma-separated columns and a decimal point for numeric values. Keep the same number of columns in each row. For PCA with a target, place the target last.')
step(2,'Import and check','Select **Dataset**, then **Import dataset**. Check the filename, row count and column names. The supplied example shows 768 rows and 9 columns. Check missing values before training.')
image(IMG/'ibex_page.png','Figure 2  The supplied energy2.csv example after import. Confirm the displayed counts before continuing.',6.8)
p('Read the correlation matrix','Heading 2')
t=doc.add_table(rows=1,cols=2);t.autofit=False;t.columns[0].width=Inches(3.35);t.columns[1].width=Inches(3.71)
c=t.cell(0,0);c.width=Inches(3.35);c.paragraphs[0].paragraph_format.line_spacing=Pt(1)
image(IMG/'chart_correlation_matrix.png','Figure 3  Correlations for the supplied dataset.',3.18,c)
c=t.cell(0,1);c.width=Inches(3.71);c.paragraphs[0].paragraph_format.line_spacing=Pt(1)
p('Red indicates positive correlation; blue indicates negative correlation. White is close to zero. The color ribbon runs from −1 to +1.',parent=c,size=10.5)
p('Here, X5 and y1 have a correlation of about 0.89. X1 and X2 are almost perfectly negatively correlated at −0.99. Strongly related inputs may contain overlapping information.',parent=c,size=10.5)
p('Correlation describes linear association, not causation. A value near zero does not rule out a nonlinear relationship. Use column statistics and the histogram to check ranges and unusual values too.',parent=c,size=10.5)

page(3,'Prepare features and apply PCA','Open **Prepare data** after checking the imported dataset. Choose the steps that the data needs. Cleaning and transformations change the active dataset used for subsequent training.')
table(['Tool','How to use it'],[
('Describe','Select Run Describe to review statistics for every column.'),
('Missing Values','Run Detect Missing. Choose Mean, Max, Min, Mode or Remove Row, then Handle Missing. Check that the choice fits your data.'),
('Anomaly Detection','Review flagged rows before enabling Remove anomalies from data. An unusual observation can still be valid.'),
('Normalize','Choose Z-Score, Min-Max or Max-Abs to rescale columns. Keep the original data and its units documented.'),
('Label Encoder','Convert categorical values to numeric labels. Keep the mapping; numeric codes do not automatically imply an ordered measurement.'),
('PCA','Replace correlated inputs with fewer components. Preserve the target when preparing regression or classification data.')],[1.48,5.58],10)
p('Choose how many PCA components to retain','Heading 2')
step(1,'Preserve the target','For this example, y1 must be last. Keep **Last column is target (Y)** checked so PCA transforms only X1 to X8.')
step(2,'Choose the mode','Select **Variance Threshold** and enter **0.95** to retain at least 95% of feature variance, or choose **Fixed Components** and enter a count. Then select **Run PCA**.')
step(3,'Check the result','Review the components retained and the cumulative variance chart. A 95% variance threshold is not 95% prediction accuracy; evaluate the trained model separately.')
image(IMG/'chart_cumulative_variance.png','Figure 4  Components retained (horizontal) versus cumulative variance in percent (vertical). The red line marks 95%.')
note('Enter original features when predicting','After training, the prediction form asks for original feature values. The model applies its saved normalization and PCA steps internally and returns the target in its original units. Save the model to retain this recipe.')
p('Validation note  Normalize and PCA are applied before the training split in the current app. K-fold validation does not refit these steps within each fold. Treat these scores as exploratory; independent evaluation is needed for a strict estimate of performance on new data.',size=9.5,color=GRAY)

page(4,'Train and evaluate a regression model','Use regression when the target is numeric. Linear Regression provides a baseline before trying more flexible methods. The supplied chart illustrates how to read a result; it is not a guaranteed outcome from the settings below.')
step(1,'Select the model','Open **Regression**, choose **Linear Regression** and select **y1** as the **Target Column**. After PCA, the model uses the retained components internally.')
step(2,'Set up the first run','Use an **80/20** train/test split with **Use K-Fold CV** off. Start with the displayed iteration and learning-rate values and **Regularization: None**. Select **Train model**.')
step(3,'Inspect the fit','Check the loss curve and held-out test results. If loss rises or becomes unstable, lower the learning rate. If it is still falling, consider more iterations. Compare test performance after each change.')
image(IMG/'chart_test_scatter.png','Figure 5  Supplied linear regression test scatter. Points near the diagonal indicate predictions close to actual values.',6.15)
table(['Metric','How to interpret it'],[
('R²','Higher is better; 1 is a perfect fit. Negative values can occur. The linear summary uses a percentage scale; the fold table uses ordinary R².'),
('MAE and RMSE','Lower is better. RMSE penalizes larger errors more strongly. If the target was normalized, training metrics remain in the transformed target scale.')],[1.25,5.81],10)
p('Validate and predict','Heading 2')
p('For a follow-up run, enable **Use K-Fold CV**, enter **5** folds and train again. The fold table shows training and test scores. **Combined** pools held-out predictions, with each row tested once; it is not the score of the final model fitted to all rows.',size=10.5)
p('In **Predict**, enter original feature values and select **Predict**. Use **Save Model** to keep the model and preprocessing recipe. **Save tables (CSV)** exports numbers; **Save charts / images** exports plots; **Save page image** captures the current visible page.',size=10.5)

for root in [doc._element,doc.styles._element]:
    for border in root.xpath('.//w:pBdr'):
        border.getparent().remove(border)
OUT.parent.mkdir(parents=True,exist_ok=True);doc.save(OUT);print(OUT)
