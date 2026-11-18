"""Build the first-phase progress report using the supplied Word report style.

Creates a working copy of the provided progress-report DOCX, retains its style
package, and replaces report body content with the local English Markdown.
Run from any directory with the project Python environment.
"""
from pathlib import Path
import json
import re
import shutil
import subprocess
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parents[1]
MD = HERE / 'report.md'
CONFIG = json.loads((HERE / 'report_config.json').read_text(encoding='utf-8'))
TEMPLATE = Path(r'C:\Users\huynh\Desktop\gen_report\mẫu báo cáo từ bạn - NguyenNgocDuy_E14_BCDK1-1.docx')
OUTPUT = HERE / 'output' / 'DentalSelfie_Progress_Report_01.docx'
FONT = 'Times New Roman'
BLUE = '365F91'
MID_BLUE = '4F81BD'
LIGHT = 'EAF1F7'
GRAY = '666666'
PAGE_W, PAGE_H = 8.27, 11.69
LEFT, RIGHT, TOP, BOTTOM = 1.18, .79, .79, .79
TEXT_W = PAGE_W - LEFT - RIGHT


def set_font(run, size=11, bold=None, italic=None, color='000000', name=FONT):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), name)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None: run.bold = bold
    if italic is not None: run.italic = italic


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn('w:shd'))
    if shd is None:
        shd = OxmlElement('w:shd'); tc_pr.append(shd)
    shd.set(qn('w:fill'), fill)


def set_cell_margins(cell, top=70, start=75, bottom=70, end=75):
    tc = cell._tc; tcPr = tc.get_or_add_tcPr()
    margins = tcPr.first_child_found_in('w:tcMar')
    if margins is None:
        margins = OxmlElement('w:tcMar'); tcPr.append(margins)
    for tag, value in [('top', top), ('start', start), ('bottom', bottom), ('end', end)]:
        node = margins.find(qn('w:'+tag))
        if node is None: node=OxmlElement('w:'+tag); margins.append(node)
        node.set(qn('w:w'),str(value));node.set(qn('w:type'),'dxa')


def set_table_borders(table, color='D5DDE5', size='5'):
    tblPr=table._tbl.tblPr
    borders=tblPr.find(qn('w:tblBorders'))
    if borders is None: borders=OxmlElement('w:tblBorders');tblPr.append(borders)
    for edge in ('top','left','bottom','right','insideH','insideV'):
        el=borders.find(qn('w:'+edge))
        if el is None:el=OxmlElement('w:'+edge);borders.append(el)
        el.set(qn('w:val'),'single');el.set(qn('w:sz'),size);el.set(qn('w:space'),'0');el.set(qn('w:color'),color)


def add_field(paragraph, instruction):
    fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),instruction)
    run=OxmlElement('w:r');text=OxmlElement('w:t');text.text='1';run.append(text);fld.append(run)
    paragraph._p.append(fld)


def add_bottom_border(paragraph, color='8095A8', size='6'):
    pPr=paragraph._p.get_or_add_pPr();pBdr=pPr.find(qn('w:pBdr'))
    if pBdr is None:pBdr=OxmlElement('w:pBdr');pPr.append(pBdr)
    edge=OxmlElement('w:bottom');edge.set(qn('w:val'),'single');edge.set(qn('w:sz'),size);edge.set(qn('w:space'),'4');edge.set(qn('w:color'),color);pBdr.append(edge)


def set_page_border(section):
    sect=section._sectPr
    borders=OxmlElement('w:pgBorders');borders.set(qn('w:offsetFrom'),'text')
    for side in ('top','left','bottom','right'):
        edge=OxmlElement('w:'+side);edge.set(qn('w:val'),'single');edge.set(qn('w:sz'),'8');edge.set(qn('w:space'),'18');edge.set(qn('w:color'),'7F8C97');borders.append(edge)
    sect.append(borders)


def set_page_numbering(section, fmt, start=1):
    sect=section._sectPr
    el=sect.find(qn('w:pgNumType'))
    if el is None:el=OxmlElement('w:pgNumType');sect.append(el)
    el.set(qn('w:fmt'),fmt);el.set(qn('w:start'),str(start))


def add_header_footer(section, label, footer_label, page_fmt=None):
    section.header.is_linked_to_previous=False
    section.footer.is_linked_to_previous=False
    hp=section.header.paragraphs[0];hp.clear();hp.alignment=WD_ALIGN_PARAGRAPH.LEFT
    hp.paragraph_format.tab_stops.add_tab_stop(Inches(TEXT_W),WD_TAB_ALIGNMENT.RIGHT)
    set_font(hp.add_run(label),9,color=GRAY)
    set_font(hp.add_run('\tBACHELOR THESIS'),9,color=GRAY)
    add_bottom_border(hp,'A6B2BD','4')
    fp=section.footer.paragraphs[0];fp.clear();fp.alignment=WD_ALIGN_PARAGRAPH.LEFT
    fp.paragraph_format.tab_stops.add_tab_stop(Inches(TEXT_W),WD_TAB_ALIGNMENT.RIGHT)
    set_font(fp.add_run(footer_label),9,color=GRAY)
    set_font(fp.add_run('\t'),9,color=GRAY)
    add_field(fp,'PAGE')
    if page_fmt:set_page_numbering(section,page_fmt,1)


def clear_header_footer(section):
    for part in [section.header,section.footer,section.first_page_header,section.first_page_footer]:
        part.is_linked_to_previous=False
        root=part._element
        for child in list(root):root.remove(child)
        part.add_paragraph()


def set_run_markup(paragraph, text, size=11, color='000000'):
    # Small Markdown inline formatter for bold, italics, and inline code.
    pattern=re.compile(r'(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)')
    cursor=0
    for match in pattern.finditer(text):
        if match.start()>cursor:set_font(paragraph.add_run(text[cursor:match.start()]),size,color=color)
        token=match.group(0)
        if token.startswith('**'):
            set_font(paragraph.add_run(token[2:-2]),size,bold=True,color=color)
        elif token.startswith('*'):
            set_font(paragraph.add_run(token[1:-1]),size,italic=True,color=color)
        else:
            set_font(paragraph.add_run(token[1:-1]),size,name='Consolas',color=color)
        cursor=match.end()
    if cursor<len(text):set_font(paragraph.add_run(text[cursor:]),size,color=color)


def add_body_paragraph(doc, text, style=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=11):
    p=doc.add_paragraph(style=style)
    p.alignment=align
    p.paragraph_format.line_spacing=1.12
    p.paragraph_format.space_after=Pt(5)
    set_run_markup(p,text,size)
    return p


def add_heading(doc,text,level,first=False):
    if level==1 and not first:
        doc.add_page_break()
    p=doc.add_paragraph(style=f'Heading {level}')
    p.paragraph_format.keep_with_next=True
    p.paragraph_format.space_before=Pt(12 if level==1 else 8)
    p.paragraph_format.space_after=Pt(5)
    if level==1:
        p.paragraph_format.page_break_before=False
    set_font(p.add_run(text),16 if level==1 else 13 if level==2 else 11,
              bold=True,italic=level==3,color=BLUE if level==1 else MID_BLUE if level==2 else '333333')
    if level==1:add_bottom_border(p,'A6B2BD','5')
    return p


def add_table(doc, rows):
    if not rows:return
    ncols=max(len(r) for r in rows)
    tbl=doc.add_table(rows=len(rows),cols=ncols)
    # Keep compact evidence tables together so a caption and its header are never orphaned.
    tblPr=tbl._tbl.tblPr
    cant_split=OxmlElement('w:cantSplit')
    tblPr.append(cant_split)
    tbl.alignment=WD_TABLE_ALIGNMENT.LEFT;tbl.autofit=False
    tbl.allow_autofit=False
    width=Inches(TEXT_W/max(1,ncols))
    for row in tbl.rows:
        row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
        for cell in row.cells:
            cell.width=width;cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER;set_cell_margins(cell)
    for i,values in enumerate(rows):
        for j in range(ncols):
            cell=tbl.cell(i,j);cell.text=''
            if i==0:shade(cell,'E6E6E6')
            p=cell.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.LEFT if ncols>4 else WD_ALIGN_PARAGRAPH.CENTER if i==0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.0
            value=values[j] if j<len(values) else ''
            set_run_markup(p,value,8.2 if ncols>=5 else 9.2)
            if i==0:
                for run in p.runs:run.bold=True
    set_table_borders(tbl)
    doc.add_paragraph().paragraph_format.space_after=Pt(2)
    return tbl


def add_picture(doc, alt, relative):
    path=HERE / relative
    if not path.is_file():raise FileNotFoundError(path)
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(2)
    pic=p.add_run().add_picture(str(path),width=Inches(TEXT_W-.15))
    max_h=Inches(4.2)
    if pic.height>max_h:
        ratio=max_h/pic.height
        pic.width=int(pic.width*ratio);pic.height=int(pic.height*ratio)
    p.paragraph_format.keep_with_next=True
    return p


def md_table_rows(lines):
    rows=[]
    for line in lines:
        vals=[x.strip() for x in line.strip().strip('|').split('|')]
        if vals and all(re.fullmatch(r':?-{3,}:?',x.replace(' ','')) for x in vals):continue
        rows.append(vals)
    return rows


def set_doc_styles(doc):
    normal=doc.styles['Normal'];normal.font.name=FONT;normal.font.size=Pt(11);normal.font.color.rgb=RGBColor(0,0,0)
    normal._element.rPr.rFonts.set(qn('w:eastAsia'),FONT)
    normal.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing=1.12;normal.paragraph_format.space_after=Pt(5)
    for name,size,color in [('Heading 1',16,BLUE),('Heading 2',13,MID_BLUE),('Heading 3',11,'333333')]:
        st=doc.styles[name];st.font.name=FONT;st.font.size=Pt(size);st.font.bold=True;st.font.color.rgb=RGBColor.from_string(color)
        st._element.rPr.rFonts.set(qn('w:eastAsia'),FONT)
        st.paragraph_format.keep_with_next=True
    # Keep all page geometry inherited from the reference's final body section.
    for section in doc.sections:
        section.page_width=Inches(PAGE_W);section.page_height=Inches(PAGE_H)
        section.left_margin=Inches(LEFT);section.right_margin=Inches(RIGHT)
        section.top_margin=Inches(TOP);section.bottom_margin=Inches(BOTTOM)


def add_cover(doc, second=False):
    logo=HERE/'references'/'ptit-logo.png'
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_before=Pt(22 if not second else 12)
    if logo.exists():p.add_run().add_picture(str(logo),height=Inches(.9 if not second else .7))
    for line in CONFIG['institution'][:3]:
        q=doc.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.space_after=Pt(2)
        set_font(q.add_run(line),11,bold=True)
    q=doc.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.space_before=Pt(18)
    set_font(q.add_run(CONFIG['report_title']),20,bold=True)
    q=doc.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER
    set_font(q.add_run(CONFIG['subtitle']),16,bold=True)
    if not second:
        q=doc.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.space_before=Pt(28);q.paragraph_format.space_after=Pt(10)
        set_font(q.add_run('Research Topic:'),12,bold=True,italic=True)
        q=doc.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.left_indent=Inches(.45);q.paragraph_format.right_indent=Inches(.45)
        set_font(q.add_run(CONFIG['research_topic'].upper()),14,bold=True)
        q=doc.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.space_before=Pt(22)
        set_font(q.add_run(f"Prototype: {CONFIG['prototype_name']}"),12,italic=True,color=GRAY)
        q=doc.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.space_before=Pt(100)
        set_font(q.add_run(CONFIG['city']+', '+CONFIG['report_date']),12,bold=True)
    else:
        q=doc.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.space_before=Pt(15)
        set_font(q.add_run(CONFIG['institution'][3]),11,bold=True)
        q=doc.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.space_before=Pt(12)
        set_font(q.add_run('THESIS PROGRESS REPORT No. 1'),14,bold=True)
        q=doc.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.left_indent=Inches(.35);q.paragraph_format.right_indent=Inches(.35)
        set_font(q.add_run(CONFIG['research_topic']),12,bold=True)
        rows=[["ROLE", "NAME", "STUDENT ID"],
              ["Academic supervisor", CONFIG['supervisor'], ""],
              *[[x['role'],x['name'],x['student_id']] for x in CONFIG['students']]]
        add_table(doc,rows)
        for label,value in [('CLASS',CONFIG['class']),('MAJOR',CONFIG['major'])]:
            q=doc.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER
            set_font(q.add_run(f'{label}: {value}'),11,bold=True)
        q=doc.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.paragraph_format.space_before=Pt(95)
        set_font(q.add_run(CONFIG['city']+', '+CONFIG['report_date']),12,bold=True)


def add_toc_page(doc):
    p=doc.add_paragraph(style='Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after=Pt(12)
    set_font(p.add_run('TABLE OF CONTENTS'),16,bold=True,color=BLUE)
    p=doc.add_paragraph();p.paragraph_format.space_before=Pt(8)
    add_field(p,'TOC \\o "1-2" \\h \\z \\u')


def add_frontmatter(doc, lines):
    # Use the report's authored frontmatter text and tables without duplicating cover data.
    start=next(i for i,l in enumerate(lines) if l.strip()=='## ACKNOWLEDGEMENTS')
    end=next(i for i,l in enumerate(lines) if l.strip()=='# INTRODUCTION')
    subset=lines[start:end]
    current=[];i=0
    while i<len(subset):
        line=subset[i].strip()
        if not line:i+=1;continue
        if line.startswith('## '):
            title=line[3:]
            p=add_heading(doc,title,1,first=True)
            if title in {'ACKNOWLEDGEMENTS','LIST OF FIGURES AND DIAGRAMS','ABBREVIATIONS','TEAM WORK PLAN'}:
                p.paragraph_format.page_break_before=True
            i+=1;continue
        if line.startswith('**Table '):
            p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before=Pt(4);p.paragraph_format.space_after=Pt(3)
            p.paragraph_format.keep_with_next=True
            set_font(p.add_run(line[2:-2]),9.5,bold=True,italic=True,color='333333')
            i+=1;continue
        if line.startswith('|'):
            block=[]
            while i<len(subset) and subset[i].strip().startswith('|'):
                block.append(subset[i].strip());i+=1
            add_table(doc,md_table_rows(block));continue
        if line.startswith('> '):line=line[2:]
        if re.match(r'^\d+\.\s',line):
            p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.LEFT
            set_run_markup(p,line);i+=1;continue
        add_body_paragraph(doc,line);i+=1


def add_main_content(doc, lines):
    start=next(i for i,l in enumerate(lines) if l.strip()=='# INTRODUCTION')
    lines=lines[start:]
    i=0;first_heading=True;figure_counter=0;image_map={}
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('```mermaid'):
            # Mermaid sources are maintained separately and rendered as explicit image entries.
            i+=1
            while i<len(lines) and lines[i].strip()!='```':i+=1
            i+=1;continue
        if line.startswith('```'):
            i+=1;code=[]
            while i<len(lines) and not lines[i].strip().startswith('```'):
                code.append(lines[i].rstrip());i+=1
            p=doc.add_paragraph();p.paragraph_format.left_indent=Inches(.2);p.paragraph_format.right_indent=Inches(.2)
            p.paragraph_format.space_after=Pt(5)
            set_font(p.add_run('\n'.join(code)),9,name='Consolas',color='333333')
            i+=1;continue
        if line.startswith('#### '):
            add_heading(doc,line[5:].strip(),3);i+=1;continue
        if line.startswith('### '):
            add_heading(doc,line[4:].strip(),2);i+=1;continue
        if line.startswith('## '):
            add_heading(doc,line[3:].strip(),1,first=False);i+=1;continue
        if line.startswith('# '):
            add_heading(doc,line[2:].strip(),1,first=first_heading);first_heading=False;i+=1;continue
        if line.startswith('**Table '):
            p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before=Pt(4);p.paragraph_format.space_after=Pt(3)
            p.paragraph_format.keep_with_next=True
            set_font(p.add_run(line[2:-2]),9.5,bold=True,italic=True,color='333333')
            i+=1;continue
        if line.startswith('|'):
            block=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                block.append(lines[i].strip());i+=1
            add_table(doc,md_table_rows(block));continue
        if line.startswith('!['):
            match=re.match(r'!\[(.*?)\]\((.*?)\)',line)
            if match:
                alt,path=match.groups();add_picture(doc,alt,path);figure_counter+=1
                i+=1
                if i<len(lines) and lines[i].strip().startswith('**Figure '):
                    cap=lines[i].strip().strip('*')
                    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(8);p.paragraph_format.keep_with_next=False
                    set_font(p.add_run(cap),9.5,italic=True,color='333333');i+=1
                continue
        if line.startswith('> '):
            p=doc.add_paragraph();p.paragraph_format.left_indent=Inches(.25);p.paragraph_format.right_indent=Inches(.2)
            p.paragraph_format.space_before=Pt(4);p.paragraph_format.space_after=Pt(7)
            set_run_markup(p,line[2:],10);i+=1;continue
        if line.startswith('---'):
            i+=1;continue
        if re.match(r'^\d+\.\s',line):
            p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent=Inches(.22);p.paragraph_format.first_line_indent=Inches(-.22)
            p.paragraph_format.space_after=Pt(3);set_run_markup(p,line);i+=1;continue
        if line.startswith('- '):
            p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent=Inches(.22);p.paragraph_format.first_line_indent=Inches(-.16)
            p.paragraph_format.space_after=Pt(3);set_run_markup(p,'• '+line[2:]);i+=1;continue
        add_body_paragraph(doc,line);i+=1


def refresh_word_fields(path):
    """Use installed Microsoft Word to refresh the dynamic TOC and page fields."""
    escaped=str(path).replace("'","''")
    ps=("$ErrorActionPreference='Stop'; $w=$null; $d=$null; "
        "try { $w=New-Object -ComObject Word.Application; $w.Visible=$false; $w.DisplayAlerts=0; "
        f"$d=$w.Documents.Open('{escaped}',$false,$false); $d.Repaginate(); [void]$d.Fields.Update(); "
        "if($d.TablesOfContents.Count -gt 0){[void]$d.TablesOfContents.Item(1).Update()}; "
        "$d.Save(); Write-Output ('Pages='+$d.ComputeStatistics(2)); $d.Close(0); $w.Quit() } "
        "finally { if($d){[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($d)}; "
        "if($w){[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($w)} }")
    result=subprocess.run(['powershell.exe','-NoProfile','-Command',ps],capture_output=True,text=True,timeout=120)
    if result.returncode:
        raise RuntimeError(f'Microsoft Word field refresh failed: {result.stderr or result.stdout}')
    print(result.stdout.strip())


def main():
    if not TEMPLATE.is_file():raise FileNotFoundError(f'Reference report missing: {TEMPLATE}')
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(TEMPLATE,OUTPUT)
    doc=Document(OUTPUT)
    body=doc._element.body
    for child in list(body):
        if child.tag!=qn('w:sectPr'):body.remove(child)
    set_doc_styles(doc)
    for sec in doc.sections:
        sec.different_first_page_header_footer=True
        set_page_border(sec)
    lines=MD.read_text(encoding='utf-8-sig').splitlines()
    add_cover(doc,False);doc.add_page_break();add_cover(doc,True)
    doc.add_section(WD_SECTION.NEW_PAGE)
    for idx,sec in enumerate(doc.sections):
        sec.different_first_page_header_footer=(idx==0)
        sec.page_width=Inches(PAGE_W);sec.page_height=Inches(PAGE_H)
        sec.left_margin=Inches(LEFT);sec.right_margin=Inches(RIGHT)
        sec.top_margin=Inches(TOP);sec.bottom_margin=Inches(BOTTOM)
        if idx>0:
            pg=sec._sectPr.find(qn('w:pgBorders'))
            if pg is not None:sec._sectPr.remove(pg)
        clear_header_footer(sec)
    add_front_section=doc.sections[1]
    front=add_front_section
    add_header_footer(front,'DENTALSELFIE | PROGRESS REPORT 01','DENTALSELFIE RESEARCH TEAM','lowerRoman')
    add_toc_page(doc);add_frontmatter(doc,lines)
    doc.add_section(WD_SECTION.NEW_PAGE)
    body_sec=doc.sections[-1]
    body_sec.different_first_page_header_footer=False
    add_header_footer(body_sec,'DENTALSELFIE | GRADUATION THESIS','DENTALSELFIE RESEARCH TEAM','decimal')
    add_main_content(doc,lines)
    settings=doc.settings._element
    upd=settings.find(qn('w:updateFields'))
    if upd is None:upd=OxmlElement('w:updateFields');settings.append(upd)
    upd.set(qn('w:val'),'true')
    doc.core_properties.title='DentalSelfie First-Phase Graduation Thesis Progress Report'
    doc.core_properties.subject=CONFIG['research_topic']
    doc.core_properties.author='DentalSelfie Research Team'
    doc.save(OUTPUT)
    refresh_word_fields(OUTPUT)
    print(f'Created {OUTPUT}')
    print(f'Markdown words: {len(MD.read_text(encoding="utf-8-sig").split())}; Word sections: {len(doc.sections)}')


if __name__=='__main__':
    main()
