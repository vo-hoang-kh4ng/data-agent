import sys

def main():
    # Read all input from standard input
    data = sys.stdin.read().split()
    
    # If input is empty, just return (though constraints guarantee valid input)
    if not data:
        return

    # Construct the 9x9 grid from the input data
    grid = []
    for i in range(9):
        # Convert the slice of data for the current row to integers
        row = [int(x) for x in data[i*9 : (i+1)*9]]
        grid.append(row)
    
    # Check condition 1: Each row contains integers 1-9 exactly once
    for r in range(9):
        if len(set(grid[r])) != 9:
            print("No")
            return

    # Check condition 2: Each column contains integers 1-9 exactly once
    for c in range(9):
        if len({grid[r][c] for r in range(9)}) != 9:
            print("No")
            return

    # Check condition 3: Each 3x3 subgrid contains integers 1-9 exactly once
    for r in range(0, 9, 3):
        for c in range(0, 9, 3):
            subgrid = {grid[r+i][c+j] for i in range(3) for j in range(3)}
            if len(subgrid) != 9:
                print("No")
                return

    # If all conditions are satisfied
    print("Yes")

if __name__ == '__main__':
    main()