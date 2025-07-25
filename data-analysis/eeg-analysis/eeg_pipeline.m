function eeg_pipeline()
% EEG_PIPELINE
% -------------------------------------------------------------------------
% • Processes EEG data by calculating spectral power for configured channel
%   groups and frequency bands.
% • Computes a single Theta/Alpha power ratio automatically.
% • Removes snippets specified in cfg.post.skipped_snippets.
% -------------------------------------------------------------------------

%% =========================================================================
%% 1. CONFIGURATION (BASE)
%% =========================================================================
cfg = struct();

% --- Path Settings --------------------------------------------------------
cfg.paths.script_dir    = fileparts(mfilename('fullpath'));
cfg.paths.raw_data      = fullfile(cfg.paths.script_dir,'EEG'); % Path to the folder containing M00 files
cfg.paths.timestamps    = fullfile(cfg.paths.script_dir,'timestamps.csv');
cfg.paths.output_csv    = fullfile(cfg.paths.script_dir,'../eeg.csv');
cfg.paths.python_script = fullfile(cfg.paths.script_dir,'../lmm.py');
cfg.paths.python_exe    = ...
    'path-to-python-vevn/.venv/bin/python';

% --- Main On/Off Switches -------------------------------------------------
cfg.do_cleaning         = false;

% --- Filter settings ------------------------------------------------------
cfg.filter.hp_cutoff    = 1.0;
cfg.filter.lp_cutoff    = [];

% --- Cleaning settings ----------------------------------------------------
cfg.clean.asr_cutoff    = 'off';
cfg.clean.chan_corr     = 'off';
cfg.clean.flatline      = Inf;

% --- Analysis frequencies -------------------------------------------------
cfg.freq.theta          = [4  7];
cfg.freq.alpha          = [8 12];

% --- SIMPLIFIED: Analysis metrics -----------------------------------------
% Define the individual power metrics. The script will automatically sum all
% 'theta' metrics and divide by the sum of all 'alpha' metrics.
%
%           <<< EDIT ONLY THIS SECTION FOR EACH RUN >>>
%
cfg.analysis.metrics = { ...
    struct('label',  'Theta_All', ...
           'band',   'theta', ...
           'channels', {{'Fz','F3','F4'}}), ...
    struct('label',  'Alpha_All', ...
           'band',   'alpha', ...
           'channels', {{'Pz','P3','P4'}}) ...
};

% --- Post‑processing ------------------------------------------------------
cfg.post.skipped_snippets   = [6 13 15];   % ← your “drop list”

% --- Variant definitions (can be left empty if not needed) ----------------
cfg.variants = { ...
    struct( ...
        'label'       , 'default_run' ... % Label is now just for console display
    ) ...
};


%% =========================================================================
%% 2. READ TIMESTAMP FILE ONCE (shared by variants)
%% =========================================================================
timestamps_T = readtable(cfg.paths.timestamps);

%% =========================================================================
%% 3. RUN EACH VARIANT
%% =========================================================================
for v = 1:numel(cfg.variants)

    % ---------- 3a. Merge base config with variant overrides --------------
    c       = cfg;
    var     = cfg.variants{v};
    c.label = var.label;
    if isfield(var,'do_cleaning'); c.do_cleaning = var.do_cleaning; end
    if isfield(var,'filter');      c.filter      = overwriteStruct(c.filter,var.filter); end
    if isfield(var,'clean');       c.clean       = overwriteStruct(c.clean,var.clean); end
    if isfield(var,'analysis');    c.analysis    = overwriteStruct(c.analysis,var.analysis); end

    fprintf('\n========================\n');
    fprintf('Running analysis: %s\n',  c.label);
    fprintf('Cleaning          : %s\n',   tf2str(c.do_cleaning));
    fprintf('========================\n');

    %% ---------------------------------------------------------------------
    %% INITIALISATION (fresh for each variant)
    %% ---------------------------------------------------------------------
    [ALLEEG,EEG,CURRENTSET,ALLCOM] = eeglab; %#ok<ASGLU>

    start_cols = find(contains(timestamps_T.Properties.VariableNames, {'StartSnippet','Start_Snippet'}));
    end_cols   = find(contains(timestamps_T.Properties.VariableNames, {'EndSnippet','End_Snippet'}));
    n_snippets = numel(start_cols);

    snippet_nums = zeros(1,n_snippets);
    for k = 1:n_snippets
        vname = timestamps_T.Properties.VariableNames{start_cols(k)};
        tok   = regexp(vname,'Start[_]?Snippet(\d+)','tokens','once');
        snippet_nums(k) = str2double(tok);
    end

    metric_labels = cellfun(@(m) m.label, c.analysis.metrics, 'UniformOutput', false);
    results_cell = [{'Participant','SnippetID'}, metric_labels{:}];
    row_idx = 2;

    %% ---------------------------------------------------------------------
    %% MAIN PROCESSING LOOP (Participant → Snippets)
    %% ---------------------------------------------------------------------
    for iP = 1:height(timestamps_T)
        participant_id = timestamps_T.Participant{iP};
        m00_file = fullfile(c.paths.raw_data, sprintf('%sprocessed.m00', (participant_id)));
        if ~isfile(m00_file)
            fprintf('[Missing] Cannot find file: %s\n', m00_file);
            continue;
        end
        fprintf('\nProcessing Participant: %s\n', participant_id);

        EEG = pop_importNihonKodenM00(m00_file);
        EEG = pop_chanedit(EEG,'lookup','standard-10-5-cap385.elp');
        chans_to_keep = find(~cellfun(@isempty,{EEG.chanlocs.X}));
        if length(chans_to_keep) < EEG.nbchan
            EEG = pop_select(EEG,'channel',chans_to_keep);
        end
        original_chanlocs = EEG.chanlocs;

        if ~isempty(c.filter.lp_cutoff) && ~isnan(c.filter.lp_cutoff)
            EEG = pop_eegfiltnew(EEG, 'locutoff',c.filter.hp_cutoff, 'hicutoff',c.filter.lp_cutoff);
        else
            EEG = pop_eegfiltnew(EEG,'locutoff',c.filter.hp_cutoff);
        end

        if c.do_cleaning
            EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion',c.clean.flatline, 'ChannelCriterion',c.clean.chan_corr, ...
                'LineNoiseCriterion','off', 'Highpass','off', 'BurstCriterion',c.clean.asr_cutoff, ...
                'WindowCriterion','off', 'BurstRejection','off', 'Distance','Euclidian');
            if EEG.nbchan < length(original_chanlocs)
                EEG = pop_interp(EEG,original_chanlocs,'spherical');
            end
        end

        all_chans       = {EEG.chanlocs.labels};
        chan_idx_finder = @(cNames) find(ismember(all_chans,cNames));

        for sIdx = 1:n_snippets
            snipID = snippet_nums(sIdx);
            start_time_str = timestamps_T{iP,start_cols(sIdx)};
            end_time_str   = timestamps_T{iP,end_cols(sIdx)};
            if any(ismissing([start_time_str,end_time_str])), continue; end
            start_sec = hhmmss2seconds(char(start_time_str));
            end_sec   = hhmmss2seconds(char(end_time_str));
            if end_sec <= start_sec, continue; end
            snippet_EEG = pop_select(EEG,'time',[start_sec,end_sec]);
            if isempty(snippet_EEG.data) || snippet_EEG.pnts < 2, continue; end

            [spec,freqs] = spectopo(snippet_EEG.data,0,snippet_EEG.srate, 'plot','off','freqrange',[1 40],'plotchan',0);
            spec = 10.^(spec/10);
            theta_idx = freqs >= c.freq.theta(1) & freqs <= c.freq.theta(2);
            alpha_idx = freqs >= c.freq.alpha(1) & freqs <= c.freq.alpha(2);
            band_power = @(chs,fidx) mean(spec(chan_idx_finder(chs),fidx), 'all','omitnan');

            current_row_data = cell(1, numel(c.analysis.metrics));
            for m_idx = 1:numel(c.analysis.metrics)
                metric = c.analysis.metrics{m_idx};
                switch lower(metric.band)
                    case 'theta', f_idx = theta_idx;
                    case 'alpha', f_idx = alpha_idx;
                    otherwise,    f_idx = [];
                end
                if ~isempty(f_idx)
                    current_row_data{m_idx} = band_power(metric.channels, f_idx);
                end
            end
            results_cell(row_idx,:) = [{participant_id, sprintf('snippet%d',snipID)}, current_row_data{:}];
            row_idx = row_idx + 1;
        end
    end % participant loop

    %% ---------------------------------------------------------------------
    %% POST‑PROCESSING & EXPORT
    %% ---------------------------------------------------------------------
    if size(results_cell,1) < 2
        fprintf('\n[Error] No data processed for run %s.\n', c.label);
        continue;
    end

    results_T = cell2table(results_cell(2:end,:), 'VariableNames',results_cell(1,:));
    results_T.ParticipantNum = cellfun(@(x) sscanf(x,'P%d'),results_T.Participant);
    results_T.SnippetNum     = cellfun(@(x) sscanf(x,'snippet%d'),results_T.SnippetID);

    skipped_sorted = sort(c.post.skipped_snippets(:)');
    results_T(ismember(results_T.SnippetNum,skipped_sorted),:) = [];
    results_T.SnippetID_new = arrayfun(@(x) x - sum(skipped_sorted < x), results_T.SnippetNum);

    % --- AUTOMATICALLY calculate Theta / Alpha ratio ---
    theta_labels = {};
    alpha_labels = {};
    for m = 1:numel(c.analysis.metrics)
        metric = c.analysis.metrics{m};
        if strcmpi(metric.band, 'theta')
            theta_labels{end+1} = metric.label; %#ok<AGROW>
        elseif strcmpi(metric.band, 'alpha')
            alpha_labels{end+1} = metric.label; %#ok<AGROW>
        end
    end

    if isempty(theta_labels) || isempty(alpha_labels)
        error('Configuration error: You must define at least one "theta" and one "alpha" metric in cfg.analysis.metrics.');
    end
    
    % Sum power across all theta metrics and all alpha metrics for each row
    theta_power_sum = sum(results_T{:, theta_labels}, 2);
    alpha_power_sum = sum(results_T{:, alpha_labels}, 2);
    
    results_T.CL = theta_power_sum ./ alpha_power_sum;
    
    % --- Select final columns & export ---
    final_cols = {'ParticipantNum','SnippetID_new','CL'};
    results_T  = results_T(:,final_cols);
    results_T  = renamevars(results_T, {'ParticipantNum','SnippetID_new'}, {'Participant','SnippetID'});
    results_T  = sortrows(results_T,{'Participant','SnippetID'});

    % Use the fixed output path from the configuration
    writetable(results_T, c.paths.output_csv);

    fprintf('\n[Done] Results for "%s" written to: %s\n\n', c.label, c.paths.output_csv);

    % ---------------------------------------------------------------------
    % (Optional) Execute Python Script
    % ---------------------------------------------------------------------
    if isfile(c.paths.python_exe) && isfile(c.paths.python_script)
        pyenv('Version',c.paths.python_exe);
        pyrun(fileread(c.paths.python_script));
    end

    % ---------------------------------------------------------------------
    %  AUTOMATIC CONSOLE SUMMARY  (filters • channels • metrics)
    % ---------------------------------------------------------------------
    %% 1) Format active filter parameters into a string
    fNames = fieldnames(c.filter);
    fPairs = cell(1, numel(fNames));
    for ii = 1:numel(fNames)
        val = c.filter.(fNames{ii});
        if isnumeric(val)
            if isempty(val) || any(isnan(val)), valStr = 'off';
            elseif numel(val) == 1,              valStr = sprintf('%.2f', val);
            else,                                valStr = sprintf('[%s]', strtrim(sprintf('%.2f ', val)));
            end
        elseif ischar(val) || isstring(val)
            valStr = char(val);
        else % structs, logicals, etc.
            valStr = '<obj>';
        end
        fPairs{ii} = sprintf('%s=%s', fNames{ii}, valStr);
    end
    filterStr = strjoin(fPairs, ', ');
    
    %% 2) Get unique channels and metrics from config and results
    % Union of all channels used in analysis.metrics
    allChans = {};
    for m = 1:numel(c.analysis.metrics)
        % Safely access and unpack channel names from each metric
        if isstruct(c.analysis.metrics{m}) && isfield(c.analysis.metrics{m}, 'channels')
            allChans = [allChans, c.analysis.metrics{m}.channels{:}]; %#ok<AGROW>
        end
    end
    chanStr = strjoin(unique(allChans,'stable'), ', ');
    
    % Output metric columns from the results table
    metricCols = setdiff(results_T.Properties.VariableNames, {'Participant','SnippetID'});
    metricStr  = strjoin(metricCols, ', ');
    metricLabel = 'Metric';
    if numel(metricCols) ~= 1, metricLabel = 'Metrics'; end
    
    %% 3) Print one-liner summary
    fprintf('[Summary] Filters: %s  |  Channels: %s  |  %s: %s\n', ...
            filterStr, chanStr, metricLabel, metricStr);
end % variant loop
end % main function

%% =========================================================================
%% LOCAL FUNCTIONS
%% =========================================================================
function secs = hhmmss2seconds(t)
p = strsplit(t,':');
switch numel(p)
    case 3, h=str2double(p{1}); m=str2double(p{2}); s=str2double(p{3});
    case 2, h=0;               m=str2double(p{1}); s=str2double(p{2});
    otherwise, secs = NaN; return;
end
secs = h*3600 + m*60 + s;
end

function sOut = overwriteStruct(sIn,sOverride)
f = fieldnames(sOverride);
for k = 1:numel(f)
    sIn.(f{k}) = sOverride.(f{k});
end
sOut = sIn;
end

function str = tf2str(tf)
if tf, str = 'ON'; else, str = 'OFF'; end
end