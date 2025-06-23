from radon.visitors import ComplexityVisitor
import os
from radon.raw import analyze
from radon.metrics import h_visit

with open('../../results/cc_loc_andfalsehalstead.csv', 'w') as res:
    res.write("file,sloc,cc,halstead_vocabulary\n")
    for root, dirs, files in os.walk('../../samples/'):
        for file in files:
            if file.endswith(".py") and 'wrap' not in file:
                full_path = os.path.join(root, file)
                with open(full_path, 'r') as f:
                    src = f.read()
                    cc = ComplexityVisitor.from_code(src, off=True).complexity
                    m = analyze(src)
                    hv = h_visit(src).total.vocabulary # The Halstead vocabulary is not computed correctly by radon
                    res.write(f"{file},{m.sloc},{cc},{hv}\n")
