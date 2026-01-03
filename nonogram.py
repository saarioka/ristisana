import sys
import random
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
    
    # sum of colored blocks. Used for checking full rows/columns
    row_sums = [sum(rc) for rc in row_clues]
    col_sums = [sum(cc) for cc in col_clues]

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
                logging.debug(f'Row {r} filled from {start_rightmost[b]} to {end_leftmost[b]}')
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
                logging.debug(f'Column {c} filled from {start_rightmost[b]} to {end_leftmost[b]}')
                for h in range(start_rightmost[b], end_leftmost[b]):
                    if grid[h,c] != 1:
                        grid[h,c] = 1
                        changed = True

    plot_nonogram(grid, title='Wiggle left-right')

    # Find blocks touching a colored edge and fill them in. Also mark the next cell as empty.

    for r in range(height):
        if full_rows[r]:
            continue

        # From first edge
        if grid[r,0] == 1:
            for b in range(row_clues[r][0]):
                if grid[r,b] != 1:
                    grid[r,b] = 1
                # mark the next one as empty if in bounds
                if b+1 < width:
                    grid[r,b+1] = -1

        # From second edge
        if grid[r,width-1] == 1:
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
            for b in range(col_clues[c][0]):
                if grid[b,c] != 1:
                    grid[b,c] = 1
                if b+1 < height:
                    grid[b+1,c] = -1

        # From second edge
        if grid[height-1,c] == 1:
            for b in range(col_clues[c][-1]):
                if grid[height-1-b,c] != 1:
                    grid[height-1-b,c] = 1
                if height-1-b-1 >= 0:
                    grid[height-1-b-1,c] = -1
    
    plot_nonogram(grid, title='Edge touching blocks')

    # check for full rows/columns
    for r in range(height):
        if not full_rows[r]:
            if np.sum(grid[r,:] == 1) == row_sums[r]:
                full_rows[r] = True
                logging.debug(f'Row {r} is now fully determined')
                for c in range(width):
                    if grid[r,c] != 1:
                        grid[r,c] = -1  # mark empty

    for c in range(width):
        if not full_cols[c]:
            if np.sum(grid[:,c] == 1) == col_sums[c]:
                full_cols[c] = True
                logging.debug(f'Column {c} is now fully determined')
                for r in range(height):
                    if grid[r,c] != 1:
                        grid[r,c] = -1  # mark empty

    plot_nonogram(grid, title='Newly filled rows and columns')

    return

    # residual sums

    # These are filled with deterministic methods
    filled_r = np.sum(grid == 1, axis=1)
    filled_c = np.sum(grid == 1, axis=0)

    grid_residual = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            if grid[r,c] == 0:
                # The residual sum at this cell (the pressure to be filled, given the mass in the row and column)
                grid_residual[r,c] = (row_sums[r] + col_sums[c]) - (filled_r[r] + filled_c[c])

    def plot_residual(g):
        plt.figure(figsize=(7,7))
        plt.imshow(g, cmap='cividis', interpolation='none')
        plt.title('Sampling prior')
        plt.xticks(np.arange(-0.5, g.shape[1], 1), [])
        plt.yticks(np.arange(-0.5, g.shape[0], 1), [])
        plt.grid(color='black', linestyle='-', linewidth=1)
        plt.colorbar()
        for x in range(-1, g.shape[1], 5):
            plt.axvline(x + 0.5, color='black', linestyle='-', linewidth=2)
        for y in range(-1, g.shape[0], 5):
            plt.axhline(y + 0.5, color='black', linestyle='-', linewidth=2)
    
    def verify(g):
        ''' Verify if the grid g satisfies the nonogram constraints '''
        # total sums
        for r in range(height):
            if np.sum(g[r,:] == 1) != row_sums[r]:
                return False

        for c in range(width):
            if np.sum(g[:,c] == 1) != col_sums[c]:
                return False
        
        # number of blocks
        def count_blocks(line):
            count = 0
            in_block = False
            for v in line:
                if v == 1:
                    if not in_block:
                        count += 1
                        in_block = True
                else:
                    in_block = False
            return count
        
        for r in range(height):
            if count_blocks(g[r,:]) != len(row_clues[r]):
                return False
        for c in range(width):
            if count_blocks(g[:,c]) != len(col_clues[c]):
                return False
        
        return True

    
    def sample(g):
        # rows
        for r in range(height):
            if full_rows[r]:
                continue
        
    done = False
    while not done:
        plot_residual(grid_residual)
        plt.show()

    return grid


def nonogram_solve2(row_clues, col_clues, plot=False):
    sum_r = [sum(rc) for rc in row_clues]
    sum_c = [sum(cc) for cc in col_clues]

    print("Row sums:", sum_r, len(sum_r))
    print("Column sums:", sum_c, len(sum_c))

    # make heatmap of sums
    height = len(row_clues)
    width = len(col_clues)
    grid = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            grid[r,c] = sum_r[r] + sum_c[c]
    
    plt.figure(figsize=(7,7))
    plt.imshow(grid, interpolation='none')
    plt.xticks(np.arange(-0.5, grid.shape[1], 1), [])
    plt.yticks(np.arange(-0.5, grid.shape[0], 1), [])
    plt.grid(color='black', linestyle='-', linewidth=1)
    # coarse grid lines every 5 cells
    for x in range(-1, grid.shape[1], 5):
        plt.axvline(x + 0.5, color='black', linestyle='-', linewidth=2)
    for y in range(-1, grid.shape[0], 5):
        plt.axhline(y + 0.5, color='black', linestyle='-', linewidth=2)
    plt.title('Row sum heatmap')
    plt.colorbar()
    #plot_nonogram(grid, title='Sum heatmap')

    def random_int_partition_with_zeros_np(total, length):
        bars = np.sort(np.random.choice(total + length - 1, length - 1, replace=False))
        parts = np.diff(np.concatenate(([-1], bars, [total + length - 1]))) - 1
        return parts.tolist()
    
    # construct a valid row/column guess using clues
    def construct_guess(clues, length, force:int=1, guess0=None):
        '''
        clues: list of block sizes
        length: total length of the line
        guess0: optional initial guess
        force: if initial guess is given, use this as a magnitude for deviating from it
        '''
        # Total length - otal length of blocks - mandatory spaces between blocks
        sum_whitespace = length - sum(clues) - (len(clues) - 1)

        if guess0 is not None:
            ind1 = random.randint(0, len(guess0)-1)
            ind2 = random.randint(0, len(guess0)-1)
            # subtract "force" from one random position and add to another
            f = max(0, min(guess0[ind1], force))
            guess0[ind1] -= f
            guess0[ind2] += f
            whitespace_guess = guess0
        else:
            # Come up with a random partition of the whitespace
            # Optional whitespace can be on both ends -> +1
            whitespace_guess = random_int_partition_with_zeros_np(sum_whitespace, len(clues) + 1)
            if sum(whitespace_guess) != sum_whitespace:
                print("Error in whitespace partitioning")

        #print(f"Whitespace guess (total {sum_whitespace}, num {len(clues)}):", whitespace_guess)
        line = []
        for i in range(len(clues)):
            for j in range(whitespace_guess[i]):
                line.append(0)

            for j in range(clues[i]):
                line.append(1)
            
            if i < len(clues) - 1:
                line.append(0)  # mandatory space after block

        # trailing whitespace if any
        for j in range(whitespace_guess[-1]):
            line.append(0)
        
        #print(line, whitespace_guess)
        
        return line, whitespace_guess
    
    def cost(g, block_count_weight=10):
        # difference in number of filled cells vs clues
        total_cost = np.abs(np.sum(g==1, axis=0) - np.array(sum_c))

        # block count difference
        n_block_diff = 0
        def count_blocks(line):
            count = 0
            in_block = False
            for v in line:
                if v == 1:
                    if not in_block:
                        count += 1
                        in_block = True
                else:
                    in_block = False
            return count

        nblocks = [count_blocks(g[:,c]) for c in range(width)]
        total_cost += block_count_weight * np.abs(np.array(nblocks) - np.array([len(cc) for cc in col_clues]))
        return total_cost
    
    # Starting point: random guess
    attempts = 1000
    lowest_cost = np.inf * np.ones(width)
    best_guess = np.zeros((height, width), dtype=int)
    for attempt in range(attempts):
        guess_grid = np.zeros((height, width), dtype=int)
        whitespace_guess = []

        # guess rows, compare to column clues
        for r in range(height):
            gg, ws = construct_guess(row_clues[r], width)
            guess_grid[r,:] = gg
            whitespace_guess.append(ws)

        latest_cost = cost(guess_grid)
        print("Attempt", attempt, "cost:", np.sum(latest_cost), 'Best so far:', np.sum(lowest_cost))

        for c in range(width):
            if latest_cost[c] < lowest_cost[c]:
                lowest_cost[c] = latest_cost[c]
                best_guess[:,c] = guess_grid[:,c]

        if np.sum(lowest_cost) == 0:
            print("Found valid solution!")
            plot_nonogram(guess_grid, title='Valid solution')
            break
    
    plot_nonogram(best_guess, title='Best guess after random sampling: cost {}'.format(np.sum(lowest_cost)))

    print("best whitespace:", best_whitespace)


    return
    
    # get the whitespace map from the best guess
    whitespaces = []
    for r in range(height):
        _, ws = construct_guess(row_clues[r], width)
        whitespaces.append(ws)
    
    print('whitespaces:', whitespaces)
    
    # fine tune best guess
    for fine in range(1000):
        for r in range(height):
            gg, ws = construct_guess(row_clues[r], width, force=1, guess0=whitespaces[r])
            guess_grid[r,:] = gg
            whitespaces[r] = ws
            if sum(gg) != sum_r[r]:
                raise ValueError("Error in fine tuning row", r)
        
        latest_cost = cost(guess_grid)
        print("Fine tuning", fine, "cost:", np.sum(latest_cost), 'Best so far:', np.sum(lowest_cost))

        for c in range(width):
            if latest_cost[c] < lowest_cost[c]:
                lowest_cost[c] = latest_cost[c]
        
        for c in range(width):
            if latest_cost[c] < lowest_cost[c]:
                lowest_cost[c] = latest_cost[c]
                best_guess[:,c] = best_guess[:,c]
                print("Found valid solution during fine tuning!")
                plot_nonogram(best_guess, title='Valid solution')
                break
    
    print("Lowest cost per row:", lowest_cost)
    plot_nonogram(best_guess, title='After fine tuning: cost {}'.format(np.sum(lowest_cost)))

    print(row_clues)
    

def nonogram_solve3(row_clues, col_clues, plot=False):
    sum_r = [sum(rc) for rc in row_clues]
    sum_c = [sum(cc) for cc in col_clues]

    nblocks_r = [len(rc) for rc in row_clues]
    nblocks_c = [len(cc) for cc in col_clues]

    print("Row sums:", sum_r, len(sum_r))
    print("Column sums:", sum_c, len(sum_c))
    print("Row block counts:", nblocks_r, len(nblocks_r))
    print("Column block counts:", nblocks_c, len(nblocks_c))

    # make heatmap of sums
    height = len(row_clues)
    width = len(col_clues)
    grid = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            grid[r,c] = sum_r[r] + sum_c[c]
    
    if plot:
        plt.figure(figsize=(7,7))
        plt.imshow(grid, interpolation='none')
        plt.xticks(np.arange(-0.5, grid.shape[1], 1), [])
        plt.yticks(np.arange(-0.5, grid.shape[0], 1), [])
        plt.grid(color='black', linestyle='-', linewidth=1)
        # coarse grid lines every 5 cells
        for x in range(-1, grid.shape[1], 5):
            plt.axvline(x + 0.5, color='black', linestyle='-', linewidth=2)
        for y in range(-1, grid.shape[0], 5):
            plt.axhline(y + 0.5, color='black', linestyle='-', linewidth=2)
        plt.title('Row sum heatmap')
        plt.colorbar()
    
    print("Total filled cells needed:", sum(sum_r), 'out of ', height*width, f'({100*sum(sum_r)/(height*width):.2f}%)')
    
    # get indices of top X highest sum cells withing grid by sorting and taking first X elements
    X = sum(sum_r)
    indices = np.dstack(np.unravel_index(np.argsort(-grid.ravel()), grid.shape))[0][:X]

    # create solution grid
    grid2 = np.zeros((height, width), dtype=int)
    for idx in indices:
        grid2[idx[0], idx[1]] = 1  # mark as filled

    if plot:
        plot_nonogram(grid2, title='Greedy highest sum selection')


    def cost(g):
        # difference in number of filled cells vs clues in both directions
        #cost_c = np.abs(np.sum(g==1, axis=0) - np.array(sum_c))
        #cost_r = np.abs(np.sum(g==1, axis=1) - np.array(sum_r))
        #total_cost = sum(cost_c) + sum(cost_r)

        cost_c = np.sum(g==1, axis=0) - np.array(sum_c)
        cost_r = np.sum(g==1, axis=1) - np.array(sum_r)
        print(cost_c, cost_r)

        # 2d grid of costs
        cost_grid = np.zeros((height, width), dtype=int)
        for r in range(height):
            for c in range(width):
                cost_grid[r,c] = cost_r[r] + cost_c[c]

        total_cost = sum(np.abs(cost_c)) + sum(np.abs(cost_r))

        # block count difference
        n_block_diff = 0
        def count_blocks(line):
            count = 0
            in_block = False
            for v in line:
                if v == 1:
                    if not in_block:
                        count += 1
                        in_block = True
                else:
                    in_block = False
            return count

        best_nblocks_r = [count_blocks(g[r,:]) for r in range(height)]
        best_nblocks_c = [count_blocks(g[:,c]) for c in range(width)]

        print("- Current block counts rows:", best_nblocks_r, len(best_nblocks_r))
        print("- Current block counts cols:", best_nblocks_c, len(best_nblocks_c))

        block_count_weight = width + height
        block_count_weight = 5
        
        # if the number of blocks are different, add a large penalty
        #total_cost += block_count_weight * np.sum(np.abs(np.array(best_nblocks_r) - np.array(nblocks_r)))
        #total_cost += block_count_weight * np.sum(np.abs(np.array(best_nblocks_c) - np.array(nblocks_c)))

        # if the determined block sizes are the same, compare the difference in size
        def get_block_sizes(line):
            sizes = []
            count = 0
            in_block = False
            for v in line:
                if v == 1:
                    count += 1
                    in_block = True
                else:
                    if in_block:
                        sizes.append(count)
                        count = 0
                        in_block = False
            if in_block:
                sizes.append(count)
            return sizes
                
        for r in range(height):
            if best_nblocks_r[r] == nblocks_r[r]:
                best_sizes = get_block_sizes(g[r,:])
                true_sizes = row_clues[r]
                for i in range(len(true_sizes)):
                    total_cost += abs(best_sizes[i] - true_sizes[i])
        
        for c in range(width):
            if best_nblocks_c[c] == nblocks_c[c]:
                best_sizes = get_block_sizes(g[:,c])
                true_sizes = col_clues[c]
                for i in range(len(true_sizes)):
                    total_cost += abs(best_sizes[i] - true_sizes[i])

        return total_cost, cost_r, cost_c
    
    def swap_optimize(grid, N=3, iterations=1000):
        ''' switch N filled cells with N empty cells and see if cost improves '''
        g = grid.copy()
        best_cost, _, _ = cost(g)
        for iteration in range(iterations):
            filled_indices = np.argwhere(g == 1)
            empty_indices = np.argwhere(g == 0)

            total_cost, cost_r, cost_c = cost(g)

            print("Total cost grid sum:", total_cost)
            print("Row costs:", cost_r)
            print("Column costs:", cost_c)

            #plt.figure(figsize=(7,7))
            #plt.imshow(cost_grid, cmap='cividis', interpolation='none')
            #plt.colorbar()
            #plt.show()

            #selected_filled = filled_indices[np.random.choice(filled_indices.shape[0], N, replace=False)]
            #selected_empty = empty_indices[np.random.choice(empty_indices.shape[0], N, replace=False)]

            # weighted random choice based on cost contribution. Choose N columns and rows with highest cost.
            #cr = np.abs(cost_r)
            #cc = np.abs(cost_c)
            #cc[cc == 0] = 1
            #cr[cr == 0] = 1

            # The most positive errors (excess of filled cells)
            cr = cost_r.copy()
            cc = cost_c.copy()
            cr[cr < 1] = 1
            cc[cc < 1] = 1

            #cr = [2**float(cr) for cr in cr]
            #cc = [2**float(cc) for cc in cc]

            row_probs = cr / np.sum(cr)
            col_probs = cc / np.sum(cc)

            #print("Row selection probabilities:", row_probs)
            #print("Column selection probabilities:", col_probs)

            selected_rows_pos = np.random.choice(height, N, replace=False, p=row_probs)
            selected_cols_pos = np.random.choice(width, N, replace=False, p=col_probs)

            #print("Selected rows, pos:", selected_rows_pos)
            #print("Selected cols, pos:", selected_cols_pos)

            # The most negative errors (deficit of filled cells)
            cr = -1*cost_r.copy()
            cc = -1*cost_c.copy()
            cr[cr < 1] = 1
            cc[cc < 1] = 1

            row_probs = cr / np.sum(cr)
            col_probs = cc / np.sum(cc)

            #print("Row selection probabilities:", row_probs)
            #print("Column selection probabilities:", col_probs)

            selected_rows_neg = np.random.choice(height, N, replace=False, p=row_probs)
            selected_cols_neg = np.random.choice(width, N, replace=False, p=col_probs)

            #print("Selected rows, neg:", selected_rows_neg)
            #print("Selected cols, neg:", selected_cols_neg)

            #g[selected_filled[:,0], selected_filled[:,1]] = 0
            #g[selected_empty[:,0], selected_empty[:,1]] = 1

            # perform the swaps from positive to negative
            for i in range(N):
                r_pos = selected_rows_pos[i]
                c_pos = selected_cols_pos[i]
                r_neg = selected_rows_neg[i]
                c_neg = selected_cols_neg[i]

                # only swap if the positions are valid
                #if g[r_pos, c_pos] == 1 and g[r_neg, c_neg] == 0:
                #    g[r_pos, c_pos] = 0
                #    g[r_neg, c_neg] = 1

            new_cost, _, _ = cost(g)
            print(f"\nIteration {iteration}, cost: {new_cost}, best cost: {best_cost}")
            if new_cost < best_cost:
                best_cost = new_cost
                print("Improved cost!")
            else:
                # revert the change
                #g[selected_filled[:,0], selected_filled[:,1]] = 1
                #g[selected_empty[:,0], selected_empty[:,1]] = 0
                for i in range(N):
                    r_pos = selected_rows_pos[i]
                    c_pos = selected_cols_pos[i]
                    r_neg = selected_rows_neg[i]
                    c_neg = selected_cols_neg[i]

                    #if g[r_pos, c_pos] == 0 and g[r_neg, c_neg] == 1:
                    #    g[r_pos, c_pos] = 1
                    #    g[r_neg, c_neg] = 0
        return g
    
    grid = swap_optimize(grid, N=2, iterations=5)
    plot_nonogram(grid, title='After swapping, N=1')

    #grid = swap_optimize(grid2, N=3, iterations=1000)
    #plot_nonogram(grid, title='After swapping, N=3')

    #grid = swap_optimize(grid, N=2, iterations=1000)
    #plot_nonogram(grid, title='After swapping, N=2')

    #grid = swap_optimize(grid, N=1, iterations=1000)
    #plot_nonogram(grid, title='After swapping, N=1')


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
        #print(clues)
    
    plot_nonogram(data, title='Ground truth')

    #solution = nonogram_solve(row_clues, col_clues, plot=True)
    #solution = nonogram_solve2(row_clues, col_clues, plot=True)
    solution = nonogram_solve3(row_clues, col_clues, plot=True)

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
