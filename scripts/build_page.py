#!/usr/bin/env python3
"""Build the independent project-page Markdown with Pandoc (no README input)."""
import argparse
import copy
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs'
READER='markdown-smart-implicit_figures'
SECTION_CLASSES={'hero','content-section','research-notes'}
PARAGRAPH_CLASSES={'hero-result','intro','eyebrow'}

def pandoc(arguments,content):
    result=subprocess.run(['pandoc',*arguments],input=content,text=True,capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr.strip())
    return result.stdout

def render_fragment(document,blocks):
    fragment={'pandoc-api-version':document['pandoc-api-version'],'meta':{},'blocks':blocks}
    return pandoc(['--from=json','--to=html5','--wrap=none'],json.dumps(fragment)).strip()

def attributes(attr):
    identifier,classes,values=attr
    pairs=([('id',identifier)] if identifier else [])+([('class',' '.join(classes))] if classes else [])+values
    return ''.join(' '+html.escape(k,quote=True)+'="'+html.escape(v,quote=True)+'"' for k,v in pairs)

def raw(value):return {'t':'RawBlock','c':['html',value]}

def strip_paragraph(value):
    match=re.fullmatch(r'<p>(.*)</p>',value,re.S)
    if not match:raise ValueError('Expected one paragraph, got: '+value[:100])
    return match.group(1)

def transform(document,blocks):
    result=[]
    for block in blocks:
        kind=block['t']
        if kind=='Div':
            attr,children=block['c'];classes=set(attr[1])
            if classes & SECTION_CLASSES:
                result += [raw('<section'+attributes(attr)+'>'),*transform(document,children),raw('</section>')]
            elif classes & PARAGRAPH_CLASSES:
                if len(children)!=1 or children[0]['t']!='Para':
                    raise ValueError('The '+', '.join(attr[1])+' block must contain one paragraph.')
                content=strip_paragraph(render_fragment(document,children))
                result.append(raw('<p'+attributes(attr)+'>'+content+'</p>'))
            else:
                block=copy.deepcopy(block);block['c'][1]=transform(document,children);result.append(block)
        elif kind=='Para' and len(block['c'])==1 and block['c'][0]['t']=='Image':
            image=block['c'][0];src=image['c'][2][0]
            parsed=urlsplit(src)
            if not parsed.scheme and not parsed.netloc:
                path=(DOCS/unquote(parsed.path)).resolve()
                path.relative_to(DOCS.resolve())
                if not path.is_file():raise FileNotFoundError('Missing image: '+src)
            content=strip_paragraph(render_fragment(document,[block]))
            result.append(raw('<figure class="paper-figure"><a class="figure-link" href="'+html.escape(src,quote=True)+'" data-lightbox>'+content+'</a></figure>'))
        elif kind=='Para' and block['c'] and block['c'][0]['t']=='RawInline' and block['c'][0]['c'][0]=='html' and (
            block['c'][0]['c'][1].startswith('<summary') or block['c'][0]['c'][1].startswith('<span class="notes-link')
        ):
            result.append(raw(strip_paragraph(render_fragment(document,[block]))))
        elif kind=='Para' and len(block['c'])==1 and block['c'][0]['t']=='Span' and 'notes-link' in block['c'][0]['c'][0][1]:
            result.append(raw(strip_paragraph(render_fragment(document,[block]))))
        elif kind=='Table':
            result += [raw('<div class="table-scroll" tabindex="0" aria-label="Results table">'),block,raw('</div>')]
        else:result.append(block)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='HTML output; default: docs/index.html')
    parser.add_argument('--site-dir',type=Path,help='Create a deployable directory containing only website assets')
    args=parser.parse_args()
    if not shutil.which('pandoc'):parser.error('Pandoc is required. Install it with: sudo apt-get install pandoc')
    source=DOCS/'page.md';template=DOCS/'template.html'
    document=json.loads(pandoc(['--from='+READER,'--to=json'],source.read_text()))
    for key in ['title','site-url','authors','resources']:
        if key not in document['meta']:parser.error('Missing page metadata: '+key)
    document['blocks']=transform(document,document['blocks'])
    page=pandoc(['--from=json','--to=html5','--standalone','--template='+str(template),'--wrap=none'],json.dumps(document))
    page='<!-- Generated from page.md by scripts/build_page.py. Edit page.md, not this file. -->\n'+page
    if args.site_dir:
        destination=args.site_dir.resolve();destination.mkdir(parents=True,exist_ok=True)
        for name in ['project.css','project.js','paper.pdf','.nojekyll']:
            shutil.copyfile(DOCS/name,destination/name)
        shutil.copytree(DOCS/'assets',destination/'assets',dirs_exist_ok=True)
        output=args.output or destination/'index.html'
    else:output=args.output or DOCS/'index.html'
    output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.tmp');temporary.write_text(page);temporary.replace(output)
    print('Built '+str(output)+' from '+str(source)+'; README.md was not used.')

if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError) as error:
        print('Build failed: '+str(error),file=sys.stderr);raise SystemExit(1)
