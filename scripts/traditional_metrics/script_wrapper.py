import os

def wrap_scripts_in_functions(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.py') and not filename.startswith('wrapped_'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                script_lines = f.readlines()

            function_name = os.path.splitext(filename)[0]
            wrapped_filename = f'wrapped_{filename}'
            wrapped_filepath = os.path.join(directory, wrapped_filename)

            with open(wrapped_filepath, 'w') as wf:
                wf.write(f'def {function_name}():\n')
                for line in script_lines:
                    if line.strip():  # Avoid indenting empty lines
                        wf.write('  ' + line)
                    else:
                        wf.write('\n')

            print(f"Wrapped {filename} into {wrapped_filename}")

# Example usage
wrap_scripts_in_functions('../samples/')
