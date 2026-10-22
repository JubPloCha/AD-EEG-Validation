function verify_eeglab_reader(jobfile)
% Read-only EEGLAB comparison; exports float32 arrays without preprocessing.
jobs = jsondecode(fileread(jobfile));
for k = 1:numel(jobs)
    [folder, name, ext] = fileparts(jobs(k).set_path);
    EEG = pop_loadset('filename', [name ext], 'filepath', folder, 'loadmode', 'all');
    fid = fopen(jobs(k).binary_path, 'w');
    fwrite(fid, single(EEG.data), 'single', 0, 'ieee-le'); fclose(fid);
    meta = struct('nbchan',EEG.nbchan,'pnts',EEG.pnts,'trials',EEG.trials,'srate',EEG.srate,'labels',{{EEG.chanlocs.labels}},'eeglab',eeg_getversion);
    meta.chanlocs = EEG.chanlocs;
    fid = fopen(jobs(k).metadata_path, 'w', 'n', 'UTF-8'); fprintf(fid, '%s', jsonencode(meta)); fclose(fid);
end
end
