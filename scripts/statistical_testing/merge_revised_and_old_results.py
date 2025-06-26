with open('../../results/revised_complexities.csv', 'r') as revised:
    with open('../../results/organizedformultilevel.csv', 'r') as old:
        with open('../../results/revised_complexities_and_manual_cl.csv', 'w') as merged:
            revised_lines = revised.readlines()
            old_lines = old.readlines()
            revised_data = {}
            merged_data = []

            for l in revised_lines[1:]:
                elems = l.strip().split(',')
                revised_data[elems[1]] = elems

            merged.write('subject,ID,rating,DD,Halstead,LOC,Cyclomatic,CCCP-PD,CCCP-MDI,DD-old,Halstead-old,LOC-old,Cyclomatic-old,CCCP-PD-old,CCCP-MPI-old\n')
            for l in old_lines[1:]:
                elems = l.strip().split(',')
                subject = elems[0]
                id = elems[1]
                rating = elems[2]
                merged.write(f'{subject},{id},{rating},{",".join((revised_data.get(id)[2:]))}\n')