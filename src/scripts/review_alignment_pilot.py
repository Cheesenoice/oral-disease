"""Local, human-operated OMNI pilot annotation UI. No labels are inferred."""
from __future__ import annotations

import csv
import os
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data/processed/alignment_omni_frontal_candidates'
QUEUE = BASE / 'review_queue.csv'
BOXES = BASE / 'boxes_template.csv'
TARGETS = ('upper_midline_diastema', 'anterior_crowding')


def read_csv(path: Path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        return reader.fieldnames, list(reader)


def atomic_csv(path: Path, fields: list[str], rows: list[dict]):
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='', dir=path.parent, delete=False) as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(file.name)
    os.replace(temp_path, path)


class ReviewApp:
    def __init__(self, window: tk.Tk):
        self.window = window
        self.queue_fields, self.rows = read_csv(QUEUE)
        self.box_fields, self.boxes = read_csv(BOXES)
        self.index = 0
        self.start = None
        self.preview_id = None
        self.photo = None
        self.scale = 1.0

        window.title('DentalSelfie - OMNI local review')
        window.geometry('1100x850')
        controls = ttk.Frame(window, padding=6)
        controls.pack(fill='x')
        self.position = ttk.Label(controls)
        self.position.grid(row=0, column=0, padx=4)
        ttk.Button(controls, text='Previous', command=lambda: self.navigate(-1)).grid(row=0, column=1)
        ttk.Button(controls, text='Next', command=lambda: self.navigate(1)).grid(row=0, column=2)
        ttk.Label(controls, text='Reviewer:').grid(row=0, column=3, padx=6)
        self.reviewer = ttk.Entry(controls, width=20)
        self.reviewer.grid(row=0, column=4)
        self.view = tk.StringVar()
        self.diastema = tk.StringVar()
        self.crowding = tk.StringVar()
        self.target = tk.StringVar(value=TARGETS[0])
        for col, (label, var, choices) in enumerate((
            ('View', self.view, ('pending', 'frontal', 'other', 'ungradable')),
            ('Midline gap', self.diastema, ('pending', 'present', 'absent', 'uncertain')),
            ('Crowding', self.crowding, ('pending', 'present', 'absent', 'uncertain')),
            ('Draw target', self.target, TARGETS),
        )):
            ttk.Label(controls, text=label).grid(row=1, column=col * 2)
            ttk.Combobox(controls, textvariable=var, values=choices, state='readonly', width=22).grid(row=1, column=col * 2 + 1)
        ttk.Label(controls, text='Notes:').grid(row=2, column=0)
        self.notes = ttk.Entry(controls, width=75)
        self.notes.grid(row=2, column=1, columnspan=6, sticky='ew')
        ttk.Button(controls, text='Save review', command=self.save).grid(row=2, column=7, padx=8)
        ttk.Button(controls, text='Delete last box', command=self.delete_last).grid(row=3, column=0, columnspan=2)
        ttk.Button(controls, text='Approve boxes', command=self.approve_boxes).grid(row=3, column=2, columnspan=2)
        ttk.Label(controls, text='Drag a box, then use another reviewer for approval.').grid(row=3, column=4, columnspan=4)
        self.canvas = tk.Canvas(window, background='#262626', cursor='cross')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<ButtonPress-1>', self.mouse_down)
        self.canvas.bind('<B1-Motion>', self.mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.mouse_up)
        self.window.bind('<Left>', lambda event: self.navigate(-1))
        self.window.bind('<Right>', lambda event: self.navigate(1))
        self.show()

    def current(self):
        return self.rows[self.index]

    def show(self):
        row = self.current()
        self.position.config(text=f'{self.index + 1}/{len(self.rows)}  {row["split"]}  {row["source_image_id"]}')
        self.view.set(row['view_review'])
        self.diastema.set(row['diastema_review'])
        self.crowding.set(row['crowding_review'])
        self.reviewer.delete(0, 'end')
        if row['reviewer'] != 'pending': self.reviewer.insert(0, row['reviewer'])
        self.notes.delete(0, 'end')
        if row['review_notes'] != 'pending': self.notes.insert(0, row['review_notes'])
        image = Image.open(ROOT / row['image_path']).convert('RGB')
        self.scale = min(1000 / image.width, 620 / image.height, 1.0)
        size = (round(image.width * self.scale), round(image.height * self.scale))
        self.photo = ImageTk.PhotoImage(image.resize(size))
        self.canvas.delete('all')
        self.canvas.config(width=size[0], height=size[1])
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        for box in self.boxes:
            if box['source_image_id'] == row['source_image_id']:
                color = '#00ff99' if box['target'] == TARGETS[0] else '#ffae00'
                self.canvas.create_rectangle(*(float(box[key]) * self.scale for key in ('x_min', 'y_min', 'x_max', 'y_max')), outline=color, width=3)
                self.canvas.create_text(float(box['x_min']) * self.scale, float(box['y_min']) * self.scale, anchor='sw', fill=color, text=f'{box["target"]} [{box["review_state"]}]')

    def navigate(self, direction):
        self.index = (self.index + direction) % len(self.rows)
        self.show()

    def mouse_down(self, event):
        self.start = (event.x, event.y)

    def mouse_drag(self, event):
        if self.start is None: return
        if self.preview_id: self.canvas.delete(self.preview_id)
        self.preview_id = self.canvas.create_rectangle(*self.start, event.x, event.y, outline='#ffffff', width=2)

    def mouse_up(self, event):
        if self.start is None: return
        x1, x2 = sorted((self.start[0], event.x))
        y1, y2 = sorted((self.start[1], event.y))
        self.start = None
        if self.preview_id: self.canvas.delete(self.preview_id)
        self.preview_id = None
        if x2 - x1 < 5 or y2 - y1 < 5: return
        reviewer = self.reviewer.get().strip()
        if not reviewer:
            messagebox.showerror('Reviewer required', 'Enter a reviewer name before drawing a box.')
            return
        width, height = int(self.current()['width']), int(self.current()['height'])
        coords = [max(0, min(limit, round(value / self.scale))) for value, limit in zip((x1, y1, x2, y2), (width, height, width, height))]
        if coords[0] >= coords[2] or coords[1] >= coords[3]: return
        self.boxes.append(dict(zip(self.box_fields, (self.current()['source_image_id'], self.target.get(), *coords, reviewer, 'pending', 'UI proposal'))))
        atomic_csv(BOXES, self.box_fields, self.boxes)
        self.show()

    def delete_last(self):
        image_id = self.current()['source_image_id']
        for idx in range(len(self.boxes) - 1, -1, -1):
            if self.boxes[idx]['source_image_id'] == image_id:
                if self.boxes[idx]['review_state'] == 'approved':
                    messagebox.showerror('Approved box', 'Edit approved boxes in a separate review pass.')
                    return
                del self.boxes[idx]
                atomic_csv(BOXES, self.box_fields, self.boxes)
                self.show()
                return

    def approve_boxes(self):
        reviewer = self.reviewer.get().strip()
        if not reviewer:
            messagebox.showerror('Reviewer required', 'Enter the approving reviewer name.')
            return
        pending = [box for box in self.boxes if box['source_image_id'] == self.current()['source_image_id'] and box['review_state'] == 'pending']
        if not pending: return
        if any(box['reviewer'] == reviewer for box in pending):
            messagebox.showerror('Independent review required', 'Use a different reviewer from the person who proposed the boxes.')
            return
        for box in pending:
            box['notes'] = f'Proposed by {box["reviewer"]}; approved by {reviewer}'
            box['reviewer'] = reviewer
            box['review_state'] = 'approved'
        atomic_csv(BOXES, self.box_fields, self.boxes)
        self.show()

    def save(self):
        row = self.current()
        reviewer = self.reviewer.get().strip()
        if any(status != 'pending' for status in (self.view.get(), self.diastema.get(), self.crowding.get())) and not reviewer:
            messagebox.showerror('Reviewer required', 'Enter a reviewer name before saving a decision.')
            return
        row.update(view_review=self.view.get(), diastema_review=self.diastema.get(), crowding_review=self.crowding.get(), reviewer=reviewer or 'pending', review_notes=self.notes.get().strip() or 'pending')
        atomic_csv(QUEUE, self.queue_fields, self.rows)
        self.navigate(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true', help='Review the earlier 36-image pilot rather than the 724-image candidate pool')
    args = parser.parse_args()
    if args.pilot:
        BASE = ROOT / 'docs/reports/alignment/frontal_pilot_16_7'
        QUEUE = BASE / 'review_queue.csv'
        BOXES = BASE / 'boxes_template.csv'
    ReviewApp(tk.Tk()).window.mainloop()
