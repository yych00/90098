from pathlib import Path
import hashlib
import json
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'Combined_Report_WS2_WS3_WS4_WS5.pdf'
GROUPS = [
    (2, 'Q-Learning and SARSA', [
        ('Environment and algorithms', 1),
        ('Task 1: CliffWalking', 2),
        ('Task 2: Experimental design', 2),
        ('Task 2 Q1: Four-algorithm comparison', 3),
        ('Task 2 Q2: Q-learning learning rate', 4),
        ('Task 2 Q3: SARSA(lambda) trace parameter', 5),
        ('Conclusion and limits', 5),
        ('Reproducibility materials', 5),
    ]),
    (3, 'Deep Q-Learning', [
        ('Task 1 Q1: CartPole environment', 1),
        ('Task 1 Q2: DQN training', 2),
        ('Task 2 Q1: Learning rate', 3),
        ('Task 2 Q2: Exploration schedules', 4),
        ('Task 2 Q3: Replay capacity and batch size', 5),
    ]),
    (4, 'Policy Gradient', [
        ('Task 1 Q1: InvertedPendulum environment', 1),
        ('Task 1 Q2: Vanilla PG, reward-to-go and baseline', 2),
        ('Task 2 Q1: Learning rate', 3),
        ('Task 2 Q2: Steps per batch', 5),
        ('Task 2 Q3: CartPole PG versus DQN', 7),
        ('Task 2 Q4: HalfCheetah controller', 8),
    ]),
    (5, 'Actor Critic and Soft Actor Critic', [
        ('Task 1 Q1: HalfCheetah environment and policies', 1),
        ('Task 1 Q2: AC/GAE training on HalfCheetah', 2),
        ('Task 1 Q3: SAC training on HalfCheetah', 3),
        ('Task 2 Q1: Effect of GAE lambda', 4),
        ('Task 2 Q2: Initial entropy temperature', 5),
        ('Task 2 Q3: Actor and critic learning rates', 7),
        ('Task 2 Q4: Replay size, batch size and learning start', 9),
        ('Supplementary temperature diagnostics', 12),
        ('References and reproducibility', 13),
    ]),
]

doc = fitz.open()
for _ in range(2):
    doc.new_page(width=595.276, height=841.89)
starts = {}
sources = []
for n, title, rows in GROUPS:
    path = ROOT / f'Workshop{n}_Final_Report_EN.pdf'
    src = fitz.open(path)
    starts[n] = len(doc)
    doc.insert_pdf(src)
    sources.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'pages': len(src), 'merged_start_page': starts[n] + 1})
    src.close()

ink = (0.12, 0.18, 0.27)
muted = (0.38, 0.43, 0.49)
blue = (0.12, 0.33, 0.52)
bookmarks = [[1, 'Contents', 1]]
for page_index in range(2):
    page = doc[page_index]
    page.insert_text((48, 48), 'ELEN90098  /  REINFORCEMENT LEARNING FOR ENGINEERING',
                     fontsize=9, color=muted)
    page.insert_text((48, 86), 'Workshops 2-5: Combined Report',
                     fontname='hebo', fontsize=21, color=ink)
    page.insert_text((48, 114), 'Contents' + (' (continued)' if page_index else ''),
                     fontname='hebo', fontsize=15, color=blue)
    page.insert_text((48, 136), 'Page numbers refer to this merged PDF. Click an entry to navigate.',
                     fontsize=9, color=muted)
    page.draw_line((48, 151), (547, 151), color=(0.78, 0.83, 0.87))
    y = 183
    for n, title, rows in GROUPS[page_index * 2:page_index * 2 + 2]:
        start = starts[n]
        chapter = f'Workshop {n}: {title}'
        page.insert_text((48, y), chapter, fontname='hebo', fontsize=12, color=blue)
        number = str(start + 1)
        page.insert_text((547 - fitz.get_text_length(number, fontname='hebo', fontsize=12), y),
                         number, fontname='hebo', fontsize=12, color=blue)
        page.insert_link({'kind': fitz.LINK_GOTO, 'from': fitz.Rect(46, y-13, 549, y+5),
                          'page': start, 'to': fitz.Point(0, 0)})
        bookmarks.append([1, chapter, start + 1])
        y += 28
        for label, original_page in rows:
            target = start + original_page - 1
            assert fitz.get_text_length(label, fontsize=10) < 440
            page.insert_text((62, y), label, fontsize=10, color=ink)
            num = str(target + 1)
            page.insert_text((547 - fitz.get_text_length(num, fontsize=10), y), num,
                             fontsize=10, color=ink)
            left = 70 + fitz.get_text_length(label, fontsize=10)
            page.draw_line((left, y-2), (520, y-2), color=(0.80, 0.83, 0.86),
                           width=0.4, dashes='[1 3] 0')
            page.insert_link({'kind': fitz.LINK_GOTO, 'from': fitz.Rect(60, y-11, 549, y+5),
                              'page': target, 'to': fitz.Point(0, 0)})
            bookmarks.append([2, label, target + 1])
            y += 24
        y += 23
    assert y < 770, y
    page.draw_line((48, 779), (547, 779), color=(0.78, 0.83, 0.87))
    page.insert_text((48, 797), 'Original report pages and their printed page numbers are retained.',
                     fontsize=8, color=muted)
    page.insert_text((527, 797), f'{page_index+1} / 2', fontsize=8, color=muted)

doc.set_toc(bookmarks)
doc.set_metadata({'title': 'ELEN90098 Combined Report - Workshops 2-5',
                  'author': 'Zhengnan Cao and Yuchen Yang',
                  'subject': 'Original workshop reports with linked contents'})
doc.save(OUT, garbage=4, deflate=True)
doc.close()

merged = fitz.open(OUT)
checks = []
for src_info in sources:
    src = fitz.open(ROOT / src_info['file'])
    start = src_info['merged_start_page'] - 1
    for i in range(len(src)):
        original, copied = src[i], merged[start+i]
        assert original.get_text() == copied.get_text()
        assert original.rect == copied.rect
        a = original.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        b = copied.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        assert a.samples == b.samples, (src_info['file'], i+1)
        checks.append({'source': src_info['file'], 'source_page': i+1,
                       'merged_page': start+i+1, 'text_and_render_identical': True})
    src.close()
assert len(merged) == 34
assert all(link['page'] >= 2 for i in range(2) for link in merged[i].get_links())
for i in range(2):
    merged[i].get_pixmap(matrix=fitz.Matrix(1.35, 1.35)).save(ROOT / 'tmp' / 'pdfs' / f'contents_{i+1}.png')
qa = {'output': OUT.name, 'pages': len(merged), 'contents_pages': 2,
      'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(), 'sources': sources,
      'bookmarks': len(merged.get_toc()), 'preservation_checks': checks}
(ROOT / 'tmp' / 'pdfs' / 'merge_qa.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in qa.items() if k not in ('sources','preservation_checks')}, indent=2))
