# Export all functions from the currently loaded Ghidra program to a jsonl file, including each function's name, entry point, and signature.
# @author Dominik Bayerl (dominik.bayerl@carissma.eu)
# @category Python 3
# @license LGPL-3.0-or-later

import gzip
import json
import os


functions = currentProgram().getFunctionManager().getFunctions(True)
listing = currentProgram().getListing()

binary_filename = os.path.basename(currentProgram().getExecutablePath())
output_dir = str(askDirectory("Output Directory", "Save"))
filepath = os.path.join(output_dir, binary_filename + ".jsonl.gz")

try:
    monitor().setMessage("Exporting functions...")
    with gzip.open(filepath, "w") as fd:
        for func in functions:
            # features: ['name', 'ret_type', 'args_type', 'inst_bytes', 'boundaries', 'num_args', 'inst_strings', 'binary_filename'],
            monitor().checkCanceled()
            monitor().setMessage(func.getName())
            instructions = list(listing.getInstructions(func.getBody(), True))
            func_data = {
                "name": func.getName(),
                "entry_point": str(func.getEntryPoint()),
                "signature": func.getPrototypeString(True, True),
                "inst_bytes": [[b & 0xFF for b in inst.getParsedBytes()] for inst in instructions],
                "inst_strings": [str(inst) for inst in instructions],
                "binary_filename": binary_filename,
                "language": str(currentProgram().getLanguageID()),
            }
            fd.write((json.dumps(func_data) + "\n").encode("utf8"))
    print("Export completed successfully: {}".format(filepath))
except Exception as e:
    print("Export failed: {}".format(str(e)))