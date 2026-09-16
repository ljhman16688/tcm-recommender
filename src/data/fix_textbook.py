# -*- coding: utf-8 -*-
"""重新解析中药学教材，提取所有489味药材"""
import re, json

import glob
filepath = glob.glob('data/中药学*.md')[0]
print('Found:', filepath)
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 找各论开始
pos = content.find('# 各论')
if pos == -1: pos = content.find('# 各 论')
if pos == -1: pos = 0
body = content[pos:]

# 按 # 或 ## 分割
sections = re.split(r'\n(?=##?\s+)', body)

herbs = []
skip_set = {
    '各论','总论','目录','前言','附药','现代研究','编写说明',
    '第一节','第二节','第三节','第四节','第五节',
    '第六章','第七章','第八章','第九章','第十章',
    '第十一章','第十二章','第十三章','第十四章',
    '第十五章','第十六章','第十七章','第十八章',
    '第十九章','第二十章','第二十一章','第二十二章',
}

for sec in sections:
    m = re.match(r'##?\s+(.+?)(?:\n|$)', sec)
    if not m: continue
    title = m.group(1).strip()

    # 过滤非药材
    if len(title) > 10 or title in skip_set: continue
    if any(kw in title for kw in ['章','节','一、','二、','三、','四、','五、','六、',
                                     '七、','八、','九、','十、','附录','第','附药',
                                     '现代','应用','用法','注意','功效','药性','性能',
                                     '发散','清热','泻下','祛风','化湿','利水','温里',
                                     '理气','消食','驱虫','止血','活血','化痰','安神',
                                     '平肝','开窍','补虚','收涩','涌吐','攻毒',
                                     '总论','各论','一（','二（','三（','四（','五（',
                                     '影响','常见','贮藏','采集','炮制','起源','发展',
                                     '一、','二、','三、','四、','五、','六、']): continue

    # 至少要有【药性】或【功效】才算是药材
    if '【药性】' not in sec and '【功效】' not in sec: continue
    nature = ''
    taste = []
    meridian = []
    toxic = False

    m2 = re.search(r'【药性】\s*(.+?)(?:\n|【|$)', sec)
    if m2:
        raw = m2.group(1).strip()
        if '小毒' in raw or '有毒' in raw or '大毒' in raw:
            toxic = True
            raw = re.sub(r'[；;]\s*有(?:小|大)?毒', '', raw)
        if '归' in raw:
            bg, ag = raw.split('归', 1)
        else:
            bg, ag = raw, ''
        bg = bg.rstrip('。；, ')
        parts = [p.strip() for p in re.split(r'[，,]', bg) if p.strip()]
        if len(parts) >= 2:
            nature_part = parts[-1]
            for kw in ['大热','大寒','微寒','微温','寒','热','温','凉','平']:
                if kw in nature_part: nature = kw; break
            for tp in parts[:-1]:
                for t in re.split(r'[、/]', tp):
                    t = t.strip()
                    if t in ['辛','甘','酸','苦','咸','淡','涩','微苦','微甘']:
                        taste.append(t)
        ag_clean = ag.replace('经','').rstrip('。；, ')
        for m_name in ['心','肝','脾','肺','肾','胃','胆','小肠','大肠','膀胱','三焦','心包']:
            if m_name in ag_clean: meridian.append(m_name)

    # 提取功效
    function = ''
    mf = re.search(r'【功效】\s*(.+?)(?:\n|【|$)', sec)
    if mf: function = mf.group(1).strip()

    # 提取使用注意
    prec = ''
    mp = re.search(r'【使用注意】\s*(.+?)(?:\n|【|$)', sec)
    if mp: prec = mp.group(1).strip()

    herbs.append({
        'name': title,
        'nature': nature,
        'taste': sorted(set(taste)),
        'meridian': meridian,
        'toxic': toxic,
        'function': function,
        'precautions': prec,
        'source': '中药学（新世纪第五版）',
    })

valid = [h for h in herbs if h['nature']]
no_nature = [h for h in herbs if not h['nature']]

print('Total herbs found: %d' % len(herbs))
print('  With nature/taste: %d' % len(valid))
print('  Missing nature: %d' % len(no_nature))
if no_nature:
    print('  Missing examples: %s' % [h['name'] for h in no_nature[:20]])

# Save
with open('data/textbook_herbs.json', 'w', encoding='utf-8') as f:
    json.dump(herbs, f, ensure_ascii=False, indent=2)
print('\nSaved %d herbs to textbook_herbs.json' % len(herbs))
