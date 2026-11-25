import { expect, test } from '@playwright/test';
import { strToU8, zipSync } from 'fflate';
const fixture = [
  {name:'k.py',mimeType:'text/plain',buffer:Buffer.from('def kernel():\n    x = 1\n    return x\n')},
  {name:'k.raw.mlir',mimeType:'text/plain',buffer:Buffer.from('%a = arith.constant 1 : i32 loc("k.py":2:5)\n')},
  {name:'k.mlir',mimeType:'text/plain',buffer:Buffer.from('%b = arith.constant 1 : i32 loc("k.py":2:5)\n')},
  {name:'k.ptx',mimeType:'text/plain',buffer:Buffer.from('.file 1 "k.py"\n.visible .entry kernel()\n{\n.loc 1 2 5\nbar.sync 0;\nmov.u32 %r1, 1;\n}\n')},
  {name:'k.sass',mimeType:'text/plain',buffer:Buffer.from('//## File "k.py", line 2\n/*0000*/ BAR.SYNC 0x0 ;\n')},
];

test('real compiler sample maps the output write across all stages on mobile', async ({page}) => {
  await page.setViewportSize({width:390,height:844});
  await page.goto('/');
  const python = page.locator('[data-stage=python]');
  await python.locator('.cm-line').filter({hasText:'out[i] = lhs + rhs'}).click();
  for (const stage of ['clean','ptx','sass'])
    await expect(page.locator(`[data-stage=${stage}] .correlation-approximate`).first()).toBeAttached();
  await page.getByRole('button',{name:'Raw IR',exact:true}).click();
  await expect(page.locator('[data-stage=raw] .correlation-approximate').first()).toBeAttached();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
});

test('sample loads, individual upload links panes, raw toggle preserves correlation and diagnostic links', async ({page}) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/');
  await expect(page.getByRole('heading',{name:'CuTe Compiler Explorer'})).toBeVisible();
  for (const stage of ['python','raw','clean','ptx','sass']) {
    if(stage==='raw') await page.getByRole('button',{name:'Raw IR',exact:true}).click();
    if(stage==='clean') await page.getByRole('button',{name:'Clean IR',exact:true}).click();
    await expect(page.locator(`[data-stage="${stage}"] .cm-content`)).not.toBeEmpty();
  }
  await page.locator('input[type=file]').setInputFiles(fixture);
  const python = page.locator('[data-stage=python]');
  await python.locator('.cm-line').filter({hasText:'x = 1'}).click();
  await expect(page.locator('[data-stage=ptx] .correlation-metadata')).toHaveCount(2);
  await expect(page.locator('[data-stage=sass] .correlation-metadata')).toHaveCount(1);
  await expect(page.locator('[data-stage=clean] .correlation-metadata')).toHaveCount(1);
  await page.getByRole('button',{name:'Raw IR',exact:true}).click();
  await expect(page.locator('[data-stage=raw] .correlation-metadata')).toHaveCount(1);
  await page.getByRole('button',{name:/synchronization/i}).first().click();
  await expect(python.locator('.correlation-metadata')).toHaveCount(1);
  expect(errors).toEqual([]);
});

test('ZIP drag drop, search, expand, resize, diagnostics collapse and theme controls', async ({page}) => {
  await page.goto('/');
  const packed = zipSync(Object.fromEntries(fixture.map(f=>[`nested/${f.name}`,strToU8(f.buffer.toString().replaceAll('"k.py"','"/original/k.py"'))])));
  const data = await page.evaluateHandle(bytes => {
    const dt = new DataTransfer(); dt.items.add(new File([Uint8Array.from(bytes)],'case.zip')); return dt;
  },Array.from(packed));
  await page.locator('main').dispatchEvent('drop',{dataTransfer:data});
  const ptx = page.locator('[data-stage=ptx]');
  await expect(ptx.getByText('k.ptx', {exact:true})).toBeVisible();
  await page.locator('[data-stage=python] .cm-line').filter({hasText:'x = 1'}).click();
  await expect(ptx.locator('.correlation-approximate')).toHaveCount(2);
  await expect(page.locator('[data-stage=sass] .correlation-approximate')).toHaveCount(1);
  await ptx.getByRole('button',{name:'Search PTX'}).click();
  await ptx.locator('input[name=search]').fill('bar.sync');
  await ptx.locator('input[name=search]').press('Enter');
  await expect(ptx.locator('.cm-searchMatch')).toHaveText('bar.sync');
  await ptx.getByRole('button',{name:'Expand PTX'}).click();
  await expect(page.locator('[data-stage=python]')).toBeHidden();
  await ptx.getByRole('button',{name:'Restore panes'}).click();
  await expect(page.locator('[data-stage=python]')).toBeVisible();
  const split = page.getByRole('separator',{name:'Resize columns'});
  const before = await page.locator('[data-stage=python]').boundingBox();
  await split.press('ArrowRight');
  await expect.poll(async () => (await page.locator('[data-stage=python]').boundingBox())!.width).toBeGreaterThan(before!.width);
  await page.getByRole('button',{name:'Hide diagnostics'}).click();
  await expect(page.getByRole('complementary',{name:'Kernel diagnostics'})).toBeHidden();
  await page.getByRole('button',{name:'Toggle theme'}).click();
  await expect(page.locator('main')).toHaveAttribute('data-theme','light');
});

test('chooses which resource report to use when multiple are loaded', async ({page}) => {
  await page.goto('/');
  await page.locator('input[type=file]').setInputFiles([
    {name:'a.log',mimeType:'text/plain',buffer:Buffer.from('ptxas info : 0 bytes spill stores, 0 bytes spill loads')},
    {name:'b.log',mimeType:'text/plain',buffer:Buffer.from('ptxas info : 32 bytes spill stores, 16 bytes spill loads')},
  ]);
  await page.getByRole('combobox',{name:'Resource report'}).selectOption('b.log');
  const finding = page.getByRole('button',{name:/Compiler report records spill traffic/});
  await expect(finding).toBeVisible();
  await finding.click();
  await expect(page.getByRole('status')).toContainText('Confirmed evidence');
  await expect(page.getByRole('region',{name:'Resource report evidence'}).locator('mark')).toContainText('32 bytes spill stores');
  await expect(page.getByRole('region',{name:'Resource report evidence'})).toBeFocused();
  await page.getByRole('combobox',{name:'Resource report'}).selectOption('a.log');
  await expect(finding).toHaveCount(0);
});

test('MLIR header controls stay clickable after an extreme column resize', async ({page}) => {
  await page.setViewportSize({width:810,height:900});
  await page.goto('/');
  const split = page.getByRole('separator',{name:'Resize columns'});
  for (let i=0; i<10; i++) await split.press('ArrowRight');
  await expect(split).toHaveAttribute('aria-valuenow','80');
  await page.getByRole('button',{name:'Raw IR',exact:true}).click({timeout:3000});
  await page.getByRole('button',{name:'Search MLIR'}).click({timeout:3000});
  await page.getByRole('button',{name:'Expand MLIR'}).click({timeout:3000});
  await expect(page.locator('[data-stage=python]')).toBeHidden();
});

test('Blackwell feedback example reveals the linked MMA instructions and wraps diagnostic text', async ({page}) => {
  await page.setViewportSize({width:810,height:900});
  await page.goto('/');
  await page.locator('input[type=file]').setInputFiles('examples/async-compute-feedback-sm100a.zip');
  await expect(page.getByText('Loaded 5 artifacts.')).toBeVisible();
  const python = page.locator('[data-stage=python]');
  await python.locator('.cm-scroller').evaluate(e => {
    e.scrollTop = 114 * e.querySelector('.cm-line')!.getBoundingClientRect().height;
  });
  await python.locator('.cm-line').filter({hasText:'ext.dot(atom, sa'}).click();
  for (const [stage, opcode] of [['clean','lir.dot'],['ptx','tcgen05.mma'],['sass','UTCHMMA']]) {
    const pane = page.locator(`[data-stage=${stage}]`);
    const lines = pane.locator('.correlation-approximate').filter({hasText:opcode});
    await expect.poll(async () => lines.evaluateAll(elements => elements.some(el => {
      const row = el.getBoundingClientRect(), scroll = el.closest('.cm-scroller')!.getBoundingClientRect();
      return row.top >= scroll.top && row.bottom <= scroll.bottom;
    }))).toBeTruthy();
  }
  const panel = page.getByRole('complementary',{name:'Kernel diagnostics'});
  await expect(panel.getByRole('button',{name:/synchronization instructions in PTX/i})).toBeVisible();
  await panel.getByText('Analysis coverage').click();
  await expect.poll(() => panel.evaluate(e => {
    const cards = [...e.querySelectorAll('.finding')];
    return e.scrollWidth <= e.clientWidth + 1 && cards.every(card => card.scrollWidth <= card.clientWidth + 1);
  })).toBeTruthy();
});

test('the selected TMA load scrolls to the bulk transfer, not its setup instructions', async ({page}) => {
  await page.setViewportSize({width:1440,height:650});
  await page.goto('/');
  await page.locator('input[type=file]').setInputFiles('examples/async-compute-feedback-sm100a.zip');
  await expect(page.getByText('Loaded 5 artifacts.')).toBeVisible();
  const python = page.locator('[data-stage=python]');
  await python.locator('.cm-scroller').evaluate(e => {
    e.scrollTop = 105 * e.querySelector('.cm-line')!.getBoundingClientRect().height;
  });
  for (const input of ['ga','gb']) {
    await python.locator('.cm-line').filter({hasText:`ext.tma_load(${input}`}).click();
    for (const [stage, opcode] of [['clean','lir.tma_load'],['ptx','cp.async.bulk.tensor'],['sass','UTMALDG']]) {
      const pane = page.locator(`[data-stage=${stage}]`);
      const lines = pane.locator('.correlation-approximate').filter({hasText:opcode});
      await expect.poll(async () => lines.evaluateAll(elements => elements.some(el => {
        const row = el.getBoundingClientRect(), scroll = el.closest('.cm-scroller')!.getBoundingClientRect();
        return row.top >= scroll.top && row.bottom <= scroll.bottom;
      }))).toBeTruthy();
    }
  }
});

test('missing metadata never fabricates a match; malformed upload is visible', async ({page}) => {
  await page.goto('/');
  await page.getByRole('button',{name:'Clear workspace'}).click();
  await page.locator('input[type=file]').setInputFiles({name:'alone.ptx',mimeType:'text/plain',buffer:Buffer.from('mov.u32 %r1, 1;')});
  await page.locator('[data-stage=ptx] .cm-line').first().click();
  await expect(page.getByRole('status')).toContainText(/no source metadata/i);
  await expect(page.getByRole('button',{name:'Raw IR',exact:true})).toBeDisabled();
  await page.locator('input[type=file]').setInputFiles({name:'bad.zip',mimeType:'application/zip',buffer:Buffer.from('bad')});
  await expect(page.getByRole('alert')).toContainText(/ZIP/i);
});
