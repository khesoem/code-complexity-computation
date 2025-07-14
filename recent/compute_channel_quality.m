function compute_channel_quality()
% COMPUTE_CHANNEL_QUALITY
% -------------------------------------------------------------------------
%  ❱ Reads timestamps.csv, crops each <participant>processed.m00 to the
%    earliest StartSnippet and latest EndSnippet.
%  ❱ Computes for every channel
%       • AvgCorr     – mean absolute Pearson r with all other channels
%       • ZCorr       – median‑based Z of AvgCorr
%       • RansacCorr  – PREP/clean_rawdata correlation with RANSAC model
%  ❱ Outputs one long table channel_quality.csv  (Participant × Channel).
%
% Designed to run on *any* MATLAB with EEGLAB; **no toolboxes required**.
% -------------------------------------------------------------------------
%% CONFIGURATION
cfg.script_dir       = fileparts(mfilename('fullpath'));
cfg.paths.raw_data   = fullfile(cfg.script_dir,'EEG');
cfg.paths.timestamps = fullfile(cfg.script_dir,'timestamps.csv');
cfg.paths.out_csv    = fullfile(cfg.script_dir,'channel_quality.csv');

cfg.hp_cutoff        = 0.50; % Hz – set [] to skip high‑pass
cfg.win_sec          = 5;    % window length for inter‑channel corr (s)

%% INITIALISE EEGLAB
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab; %#ok<ASGLU>

%% READ TIMESTAMPS
Tstamp    = readtable(cfg.paths.timestamps);
startCols = find(contains(Tstamp.Properties.VariableNames,'StartSnippet'));
endCols   = find(contains(Tstamp.Properties.VariableNames,  'EndSnippet'));

%% PREPARE RESULT CONTAINER
results = { ...
    'Participant','Channel','ChanIdx','AvgCorr','ZCorr','RansacCorr'};

%% PARTICIPANT LOOP
for iP = 1:height(Tstamp)
    pid = Tstamp.Participant{iP};
    m00 = fullfile(cfg.paths.raw_data, sprintf('%sprocessed.m00', lower(pid)));

    if ~isfile(m00)
        fprintf('[Skip] %s not found.\n', m00);
        continue;
    end

    % ---------------------------------------------------------------------
    % Parse earliest start / latest end (robust to duration or char)
    % ---------------------------------------------------------------------
    startsSec = cellfun(@val2sec, table2cell(Tstamp(iP,startCols)));
    endsSec   = cellfun(@val2sec, table2cell(Tstamp(iP,endCols)));
    validSeg  = ~isnan(startsSec) & ~isnan(endsSec);

    if ~any(validSeg)
        fprintf('[Skip] No valid snippets for %s.\n', pid);
        continue;
    end
    t0 = min(startsSec(validSeg));
    t1 = max( endsSec(validSeg));

    % ---------------------------------------------------------------------
    % Load EEG, crop, basic prep
    % ---------------------------------------------------------------------
    EEG = pop_importNihonKodenM00(m00);
    EEG = pop_chanedit(EEG,'lookup','standard-10-5-cap385.elp');
    EEG = pop_select(EEG,'channel',find(~cellfun(@isempty,{EEG.chanlocs.X})));
    EEG = pop_select(EEG,'time',[t0 t1]);

    if isempty(EEG.data)
        fprintf('[Skip] Cropped data empty for %s.\n', pid);
        continue;
    end

    if ~isempty(cfg.hp_cutoff)
        EEG = pop_eegfiltnew(EEG,'locutoff',cfg.hp_cutoff);
    end

    % ---------------------------------------------------------------------
    % Inter‑channel average correlation
    % ---------------------------------------------------------------------
    winSamp = max(1, round(cfg.win_sec * EEG.srate));
    nWin    = floor(EEG.pnts / winSamp);
    avgCorr = zeros(EEG.nbchan,1);

    for w = 1:nWin
        idx = (w-1)*winSamp + (1:winSamp);
        C   = abs(corrcoef(double(EEG.data(:,idx))')); % chan×chan
        C(1:EEG.nbchan+1:end) = NaN;                   % delete diagonal
        avgCorr = avgCorr + mean_omitnan(C,2);
    end
    avgCorr = avgCorr / nWin;

    % robust Z: median / MAD (raw)
    medCorr   = median_omitnan(avgCorr);
    madCorr   = mad_raw(avgCorr);           % raw MAD (no 1.4826 factor)
    if madCorr == 0 || isnan(madCorr), madCorr = eps; end
    zCorr     = (avgCorr - medCorr) ./ madCorr;

    % ---------------------------------------------------------------------
    % PREP / RANSAC correlation --- [UPDATED BLOCK]
    % ---------------------------------------------------------------------
    prepPar = struct('referenceChannels',1:EEG.nbchan,...
                     'evaluationChannels',1:EEG.nbchan,...
                     'ransacChannels',1:EEG.nbchan,...
                     'doInterpolate',false,...
                     'ignoreBoundaryEvents',true);
% ---------------------------------------------------------------------
% PREP / RANSAC correlation --- [FINAL CORRECTED BLOCK]
% ---------------------------------------------------------------------
prepPar = struct('referenceChannels',1:EEG.nbchan,...
                 'evaluationChannels',1:EEG.nbchan,...
                 'ransacChannels',1:EEG.nbchan,...
                 'doInterpolate',false,...
                 'ignoreBoundaryEvents',true);
    try
        det = findNoisyChannels(EEG, prepPar);
    
        % The output field is .ransacCorrelations, which is a matrix.
        % We take the median across windows (dim 2) to get one value per channel.
        ransacCorr = median(det.ransacCorrelations, 2);
    
    catch ME
        warning('PREP failed for %s: %s', pid, ME.message);
        fprintf(2, '\n--- DEBUG: Full Error Stack ---\n');
        % Loop through the error stack and print each level
        for k = 1:length(ME.stack)
            fprintf('Error in ==> %s\n       at line %d\n', ME.stack(k).file, ME.stack(k).line);
        end
        fprintf(2, '--- End of Error Stack ---\n\n');
        ransacCorr = nan(EEG.nbchan,1);
    end

    % ---------------------------------------------------------------------
    % Append participant rows
    % ---------------------------------------------------------------------
    for ch = 1:EEG.nbchan
        results(end+1,:) = { ...
            pid, EEG.chanlocs(ch).labels, ch, ...
            avgCorr(ch), zCorr(ch), ransacCorr(ch)}; %#ok<AGROW>
    end

    fprintf('[Done] %-8s | %2d ch | %3d windows\n', pid, EEG.nbchan, nWin);
end  % participant loop

%% WRITE OUTPUT
if size(results,1) < 2
    error('No data processed – check paths & timestamps.');
end
resT = cell2table(results(2:end,:), 'VariableNames', results(1,:));
writetable(resT, cfg.paths.out_csv);
fprintf('\nChannel‑quality table written to:\n  %s\n', cfg.paths.out_csv);
end  % function compute_channel_quality
% =========================================================================
% LOCAL HELPER FUNCTIONS (toolbox‑free)
% =========================================================================
function secs = hhmmss2seconds(t)
% Convert 'HH:MM:SS' or 'MM:SS' char/string → seconds (double)
    if isstring(t), t = char(t); end
    p = strsplit(t,':');
    switch numel(p)
        case 3, h=str2double(p{1}); m=str2double(p{2}); s=str2double(p{3});
        case 2, h=0;               m=str2double(p{1}); s=str2double(p{2});
        otherwise, secs = NaN; return;
    end
    secs = h*3600 + m*60 + s;
end

function s = val2sec(x)
% Generic table value → seconds (double), NaN if missing/invalid
    if ismissing(x) || (iscell(x) && isempty(x))
        s = NaN;
    elseif isduration(x)
        s = seconds(x);
    elseif isstring(x) || ischar(x)
        s = hhmmss2seconds(x);
    else
        s = NaN;
    end
end

function m = mean_omitnan(X,dim)
% Mean along DIM ignoring NaNs (toolbox‑free)
    if nargin < 2, dim = 1; end
    nanMask = isnan(X);
    cnt = size(X,dim) - sum(nanMask,dim);
    X(nanMask) = 0;
    m = sum(X,dim) ./ cnt;
    m(cnt==0) = NaN;
end

function med = median_omitnan(v)
% Median ignoring NaNs (vector)
    v = v(~isnan(v));
    if isempty(v)
        med = NaN;
    else
        med = median(v);
    end
end

function md = mad_raw(v)
% Raw median absolute deviation (no scaling), NaNs ignored
    v = v(~isnan(v));
    if isempty(v)
        md = NaN;
    else
        md = median(abs(v - median(v)));
    end
end