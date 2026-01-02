import sys
import logging
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPdfWriter, QPageSize, QPageLayout
from PyQt6.QtCore import Qt, QRectF, QSizeF, QMarginsF


def save_nonogram_as_pdf(widget, filename="nonogram.pdf", resolution=300):
    pdf_writer = QPdfWriter(filename, )
    pdf_writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))

    pdf_writer.setResolution(resolution)

    painter = QPainter(pdf_writer)

    painter.scale(2, 2)

    widget.render(painter)  # Render the widget onto the PDF
    painter.end()
    print(f"Saved Nonogram as PDF: {filename}")


class NonogramWidget(QWidget):
    def __init__(self, grid, row_clues, col_clues, cell_size=20, show_solution=False, title='Nonogram', parent=None):
        super().__init__(parent)
        self.grid = grid
        self.row_clues = row_clues
        self.col_clues = col_clues

        self.cell_size = cell_size

        self.show_solution = show_solution
        self.title = title

        self.top_margin = max(len(clues) for clues in col_clues) * cell_size + 5
        self.left_margin = max(len(clues) for clues in row_clues) * cell_size + 5

        self.setMinimumSize(
            self.left_margin + grid.shape[1]*cell_size + 50,
            self.top_margin + grid.shape[0]*cell_size + 50
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background
        painter.fillRect(self.rect(), Qt.GlobalColor.white)

        if self.show_solution:
            # Draw filled cells
            for i in range(self.grid.shape[0]):
                for j in range(self.grid.shape[1]):
                    x = self.left_margin + j*self.cell_size
                    y = self.top_margin + i*self.cell_size
                    if self.grid[i,j]:
                        painter.fillRect(x, y, self.cell_size, self.cell_size, QColor(200,150,120))

        # Draw grid lines
        pen_thin = QPen(Qt.GlobalColor.black, 1)
        pen_thick = QPen(Qt.GlobalColor.black, 2)
        for i in range(self.grid.shape[0]+1):
            y = self.top_margin + i*self.cell_size
            left_extent = self.cell_size // 2
            if i % 5 == 0 and i != self.grid.shape[0]:
                pen = pen_thick
            else:
                pen = pen_thin
            painter.setPen(pen)
            painter.drawLine(left_extent, y, self.left_margin + self.grid.shape[1]*self.cell_size, y)
        for j in range(self.grid.shape[1]+1):
            x = self.left_margin + j*self.cell_size
            top_extent = self.cell_size // 2
            if j % 5 == 0 and j != self.grid.shape[1]:
                pen = pen_thick
            else:
                pen = pen_thin
            painter.setPen(pen)
            painter.drawLine(x, top_extent, x, self.top_margin + self.grid.shape[0]*self.cell_size)
        
        # Draw clues
        painter.setPen(Qt.GlobalColor.black)
        
        # align center
        font = QFont("Arial", int(self.cell_size*0.6), QFont.Weight.Bold)
        painter.setFont(font)

        # Column clues
        for j, clues in enumerate(self.col_clues):
            x = self.left_margin + j*self.cell_size + self.cell_size/2
            for k, val in enumerate(reversed(clues)):
                y = self.top_margin - (k+1)*self.cell_size + self.cell_size/2
                rect = QRectF(self.left_margin + j*self.cell_size, 
                                    self.top_margin - (k+1)*self.cell_size, 
                                    self.cell_size, self.cell_size)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(val))

        # Row clues
        for i, clues in enumerate(self.row_clues):
            y = self.top_margin + i*self.cell_size
            for k, val in enumerate(reversed(clues)):
                rect = QRectF(self.left_margin - (k+1)*self.cell_size, 
                                    self.top_margin + i*self.cell_size, 
                                    self.cell_size, self.cell_size)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(val))
        
        bigfont = QFont("Arial", int(self.cell_size*0.7))
        painter.setFont(bigfont)
        painter.drawText(10, 100, self.title)


class NonogramWindow(QMainWindow):
    def __init__(self, grid, row_clues, col_clues, cell_size=20, show_solution=False, title='Nonogram'):
        super().__init__()
        self.setWindowTitle(title)

        self.nonogramwidget = NonogramWidget(grid, row_clues, col_clues, cell_size=cell_size, show_solution=show_solution, title=title)
        self.nonogramwidget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.nonogramwidget.setStyleSheet("background: transparent;")  # <-- remove black box

        self.setCentralWidget(self.nonogramwidget)
        self.resize(800, 800)


def nonogram_clues(matrix):
    # Generate row and column clues
    def line_clues(line):
        clues = []
        count = 0
        for v in line:
            if v == 1:
                count += 1
            elif count:
                clues.append(count)
                count = 0
        if count:
            clues.append(count)
        return clues or [0]

    rows = [line_clues(r) for r in matrix]
    cols = [line_clues(c) for c in matrix.T]
    return rows, cols


def plot_nonogram(grid, title='Nonogram'):
    plt.figure(figsize=(7,7))
    plt.imshow(grid, cmap='RdYlBu', interpolation='none')
    plt.xticks(np.arange(-0.5, grid.shape[1], 1), [])
    plt.yticks(np.arange(-0.5, grid.shape[0], 1), [])
    plt.grid(color='black', linestyle='-', linewidth=1)
    # coarse grid lines every 5 cells
    for x in range(-1, grid.shape[1], 5):
        plt.axvline(x + 0.5, color='black', linestyle='-', linewidth=2)
    for y in range(-1, grid.shape[0], 5):
        plt.axhline(y + 0.5, color='black', linestyle='-', linewidth=2)
    plt.title(title)


def nonogram_solve(row_clues, col_clues, plot=False):
    '''
    space = mandatory white cell between blocks
    gap = optional white cell between blocks or at edges
    '''
    height = len(row_clues)
    width = len(col_clues)

    # number of locations for white (including edges)
    Ns = [len(rc)+1 for rc in row_clues]
    Ms = [len(cc)+1 for cc in col_clues]

    # number of spaces between the blocks (do not count edges)
    ns = [n-2 for n in Ns]
    ms = [m-2 for m in Ms]

    logging.debug("Row spaces inside:     %s", ns)
    logging.debug("Possible column spaces: %s", Ns)

    logging.debug("Column spaces inside:  %s", ms)
    logging.debug("Possible column spaces: %s", Ms)
    full_rows = np.zeros(height, dtype=bool)
    full_cols = np.zeros(width, dtype=bool)

    for r in range(height):
        full_rows[r] = sum(row_clues[r]) + ns[r] == width
    
    for c in range(width):
        full_cols[c] = sum(col_clues[c]) + ms[c] == height
    
    logging.debug("Fully determined rows:   %s", full_rows)
    logging.debug("Fully determined columns: %s", full_cols)

    if plot:
        filled_color = 1
        empty_color = -1
        # 2d pixel map of the nonogram with the grid shown
        grid = np.zeros((height, width), dtype=int)

        # Fill in fully determined rows and columns in gray
        for r in range(height):
            if full_rows[r]:
                idx = 0
                for block in row_clues[r]:
                    grid[r, idx:idx+block] = filled_color
                    idx += block + 1  # +1 for the space
                    if idx-1 < width:
                        grid[r, idx-1] = empty_color

        for c in range(width):
            if full_cols[c]:
                idx = 0
                for block in col_clues[c]:
                    grid[idx:idx+block, c] = filled_color
                    idx += block + 1  # +1 for the space
                    if idx-1 < height:
                        grid[idx-1, c] = empty_color
    
    # Calculate permutations of spaces for rows and columns
    # permuations = size of other dimension - sum of blocks - number of spaces
    p_h = np.zeros(height, dtype=int)
    for r in range(height):
        p_h[r] = width - sum(row_clues[r]) - ns[r]
    logging.debug("Row permutations: %s", p_h)
    
    p_w = np.zeros(width, dtype=int)
    for c in range(width):
        p_w[c] = height - sum(col_clues[c]) - ms[c]
    logging.debug("Column permutations: %s", p_w)

    plot_nonogram(grid, title='Trivial rows and columns')

    # Color tiles based on "leftmost" and "rightmost" placements of blocks
    # for each row and column, place blocks as far left as possible, then as far right as possible
    # any cells that are filled in both placements are part of the solution
    # repeat until no more cells can be filled
    # (this is a simple solving technique and may not solve all nonograms)
    changed = True
    while changed:
        changed = False

        # Rows
        for r in range(height):
            if full_rows[r]:
                continue
            
            end_leftmost = []
            start_rightmost = []

            # Leftmost placement
            leftmost = np.zeros(width, dtype=int)
            idx = 0
            for b_i, block in enumerate(row_clues[r]):
                for b in range(block):
                    leftmost[idx+b] = 1
                #print(f'Start index {b_i} (left)', idx+block)
                end_leftmost.append(idx+block)
                idx += block + 1  # +1 for space

            # Rightmost placement
            rightmost = np.zeros(width, dtype=int)
            idx = width
            for b_i, block in enumerate(reversed(row_clues[r])):
                for b in range(block):
                    rightmost[idx-block+b] = 1
                #print(f'Start index {b_i} (right)', idx-block)
                start_rightmost.append(idx-block)
                idx -= block + 1  # +1 for space
            
            start_rightmost.reverse()
            #print(r, 'Leftmost block ends:', end_leftmost, 'Rightmost block starts:', start_rightmost)

            # Cells filled in both placements are part of the solution
            for b in range(len(start_rightmost)):
                if end_leftmost[b] <= start_rightmost[b]:
                    continue
                print(f'Row {r} filled from {start_rightmost[b]} to {end_leftmost[b]}')
                for w in range(start_rightmost[b], end_leftmost[b]):
                    if grid[r,w] != 1:
                        grid[r,w] = 1
                        changed = True
        
        # Columns
        for c in range(width):
            if full_cols[c]:
                continue
            
            end_leftmost = []
            start_rightmost = []

            # Leftmost placement
            leftmost = np.zeros(height, dtype=int)
            idx = 0
            for b_i, block in enumerate(col_clues[c]):
                for b in range(block):
                    leftmost[idx+b] = 1
                end_leftmost.append(idx+block)
                idx += block + 1  # +1 for space

            # Rightmost placement
            rightmost = np.zeros(height, dtype=int)
            idx = height
            for b_i, block in enumerate(reversed(col_clues[c])):
                for b in range(block):
                    rightmost[idx-block+b] = 1
                start_rightmost.append(idx-block)
                idx -= block + 1  # +1 for space
            
            start_rightmost.reverse()

            # Cells filled in both placements are part of the solution
            for b in range(len(start_rightmost)):
                if end_leftmost[b] <= start_rightmost[b]:
                    continue
                print(f'Column {c} filled from {start_rightmost[b]} to {end_leftmost[b]}')
                for h in range(start_rightmost[b], end_leftmost[b]):
                    if grid[h,c] != 1:
                        grid[h,c] = 1
                        changed = True

    plot_nonogram(grid, title='Wiggle left-right')

    # Find blocks touching a colored edge and fill them in

    for r in range(height):
        if full_rows[r]:
            continue

        # From first edge
        if grid[r,0] == 1:
            print(f'Row {r} block 0 touches first edge, filling')
            for b in range(row_clues[r][0]):
                if grid[r,b] != 1:
                    grid[r,b] = 1
                # mark the next one as empty if in bounds
                if b+1 < width:
                    grid[r,b+1] = -1

        # From second edge
        if grid[r,width-1] == 1:
            print(f'Row {r} block {len(row_clues[r])-1} touches second edge, filling')
            for b in range(row_clues[r][-1]):
                if grid[r,width-1-b] != 1:
                    grid[r,width-1-b] = 1
                if width-1-b-1 >= 0:
                    grid[r,width-1-b-1] = -1

    for c in range(width):
        if full_cols[c]:
            continue

        # From first edge
        if grid[0,c] == 1:
            print(f'Column {c} block 0 touches first edge, filling')
            for b in range(col_clues[c][0]):
                if grid[b,c] != 1:
                    grid[b,c] = 1
                if b+1 < height:
                    grid[b+1,c] = -1

        # From second edge
        if grid[height-1,c] == 1:
            print(f'Column {c} block {len(col_clues[c])-1} touches second edge, filling')
            for b in range(col_clues[c][-1]):
                if grid[height-1-b,c] != 1:
                    grid[height-1-b,c] = 1
                if height-1-b-1 >= 0:
                    grid[height-1-b-1,c] = -1

    plot_nonogram(grid, title='Edge touching blocks')

    return grid


def main():
    parser = ArgumentParser(description="Load and display data from a text file.")
    parser.add_argument('filename', type=str, help='Path to the data file')
    parser.add_argument('--cell-size', type=int, default=20, help='Size of each cell in the grid')
    parser.add_argument('--solution', action='store_true', help='Display the solution')
    parser.add_argument('-s', '--save', action='store_true', help='Save the nonogram as a PDF file')
    parser.add_argument('-t', '--title', type=str, default='Nonogram', help='Title of the nonogram, used for PDF filename also if saving')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        # disable matplotlib logging
        logging.getLogger('matplotlib').setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    data = np.loadtxt(args.filename, delimiter=',', dtype=int)
    logging.debug(f'Loaded data shape: {data.shape}')

    row_clues, col_clues = nonogram_clues(data)
    print(len(row_clues), 'rows,', len(col_clues), 'columns')

    logging.debug("Row clues:")
    for clues in row_clues:
        logging.debug(clues)

    logging.debug("Column clues:")
    for clues in col_clues:
        logging.debug(clues)

    solution = nonogram_solve(row_clues, col_clues, plot=True)

    plt.show()
    exit()


    app = QApplication(sys.argv)
    win = NonogramWindow(data, row_clues, col_clues, cell_size=args.cell_size, show_solution=args.solution, title=args.title)
    win.show()

    if args.save:
        if not args.title:
            print("Error: Title is required when saving the nonogram.")
            sys.exit(1)
        s = args.title.replace(' ', '_').lower()
        outfile_name =  s + '_solution.pdf' if args.solution else s + '.pdf'
        save_nonogram_as_pdf(win.centralWidget(), filename=outfile_name)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
