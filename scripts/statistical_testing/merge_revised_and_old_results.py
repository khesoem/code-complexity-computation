with open('../../results/backup_result_files/revised_complexities.csv', 'r') as revised:
    with open('../../results/backup_result_files/organizedformultilevel.csv', 'r') as old:
        with open('../../results/backup_result_files/revised_complexities_and_manual_cl.csv', 'w') as merged:
            revised_lines = revised.readlines()
            old_lines = old.readlines()
            revised_data = {}
            merged_data = []

            for l in revised_lines[1:]:
                elems = l.strip().split(',')
                revised_data[elems[1]] = elems

            merged.write('subject,ID,rating,DD,Halstead,LOC,Cyclomatic,CCCPPD,CCCPMDI,DDold,Halsteadold,LOCold,Cyclomaticold,CCCPPDold,CCCPMPIold\n')
            for l in old_lines[1:]:
                elems = l.strip().split(',')
                subject = elems[0]
                id = elems[1]
                rating = elems[2]
                merged.write(f'{subject},{id},{rating},{",".join((revised_data.get(id)[2:]))}\n')

with open('../../results/backup_result_files/revised_complexities.csv', 'r') as revised:
    with open('../../results/backup_result_files/old_eeg_and_complexity_results.csv', 'r') as old:
        with open('../../results/backup_result_files/revised_complexities_and_eeg_cl.csv', 'w') as merged:
            revised_lines = revised.readlines()
            old_lines = old.readlines()
            revised_data = {}
            merged_data = []

            for l in revised_lines[1:]:
                elems = l.strip().split(',')
                revised_data[elems[1]] = elems

            merged.write('subject,ID,rating,DD,Halstead,LOC,Cyclomatic,CCCPPD,CCCPMDI,DDold,Halsteadold,LOCold,Cyclomaticold,CCCPPDold,CCCPMPIold\n')
            for l in old_lines[1:]:
                elems = l.strip().split(',')
                subject = str(int(elems[0]) + 1)
                id = str(int(elems[1]) + 1)
                rating = elems[2]
                merged.write(f'{subject},{id},{rating},{",".join((revised_data.get(id)[2:]))}\n')