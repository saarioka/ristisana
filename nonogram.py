import sys
import numpy as np
from argparse import ArgumentParser

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
from PyQt6.QtCore import Qt, QRectF

class NonogramWidget(QWidget):
    def __init__(self, grid, row_clues, col_clues, cell_size=20, show_solution=False, parent=None):
        super().__init__(parent)
        self.grid = grid
        self.row_clues = row_clues
        self.col_clues = col_clues

        self.cell_size = cell_size

        self.show_solution = show_solution

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
        pen_thick = QPen(Qt.GlobalColor.black, 3)
        for i in range(self.grid.shape[0]+1):
            y = self.top_margin + i*self.cell_size
            if i % 5 == 0:
                left_extent = 0
                pen = pen_thick
            else:
                left_extent = self.left_margin
                pen = pen_thin
            painter.setPen(pen)
            painter.drawLine(left_extent, y, self.left_margin + self.grid.shape[1]*self.cell_size, y)
        for j in range(self.grid.shape[1]+1):
            x = self.left_margin + j*self.cell_size
            if j % 5 == 0:
                top_extent = 0
                pen = pen_thick
            else:
                top_extent = self.top_margin
                pen = pen_thin
            pen = pen_thick if j % 5 == 0 else pen_thin
            painter.setPen(pen)
            painter.drawLine(x, top_extent, x, self.top_margin + self.grid.shape[0]*self.cell_size)

        # Draw clues
        painter.setPen(Qt.GlobalColor.black)
        
        # align center
        font = QFont("Arial", int(self.cell_size*0.5), )
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


class NonogramWindow(QMainWindow):
    def __init__(self, grid, row_clues, col_clues, cell_size=20, show_solution=False):
        super().__init__()
        self.setWindowTitle("Nonogram Viewer")
        self.setCentralWidget(NonogramWidget(grid, row_clues, col_clues, cell_size=cell_size, show_solution=show_solution))
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


def main():
    parser = ArgumentParser(description="Load and display data from a text file.")
    parser.add_argument('filename', type=str, help='Path to the data file')
    parser.add_argument('--cell-size', type=int, default=20, help='Size of each cell in the grid')
    parser.add_argument('--solution', action='store_true', help='Display the solution')
    args = parser.parse_args()

    data = np.loadtxt(args.filename, delimiter=',', dtype=int)
    print(f'Loaded data shape: {data.shape}')

    row_clues, col_clues = nonogram_clues(data)

    print("Row clues:")
    for clues in row_clues:
        print(clues)

    print("Column clues:")
    for clues in col_clues:
        print(clues)

    app = QApplication(sys.argv)
    win = NonogramWindow(data, row_clues, col_clues, cell_size=args.cell_size, show_solution=args.solution)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
