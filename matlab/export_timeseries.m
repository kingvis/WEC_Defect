function export_timeseries(output, faultName, label, severity, Hs, Tp, runID)
% EXPORT_TIMESERIES  Write one tidy CSV per WEC-Sim run into data/raw/.
%
%   export_timeseries(output, faultName, label, severity, Hs, Tp, runID)
%
% Channels logged (consumed by the Python ML pipeline, see src/windowing.py CHANNELS):
%   t, z1, z2, rel, vrel, Fpto, Ppto, Tmoor
%
% Filename encodes metadata so windowing.py can parse labels back out:
%   run_00042__pto_damping_loss_sev60_Hs150_Tp080.csv

    t    = output.bodies(1).time;
    z1   = output.bodies(1).position(:,3);          % float heave
    z2   = output.bodies(2).position(:,3);          % spar heave
    rel  = z1 - z2;                                 % relative heave (drives PTO)
    vrel = [0; diff(rel)] ./ [1; diff(t)];          % relative velocity
    Fpto = output.ptos(1).forceTotal(:,3);          % PTO force (heave)
    Ppto = output.ptos(1).power(:,3);               % PTO instantaneous power

    % Mooring tension (heave component). Field name varies by mooring type; guard it.
    if isfield(output, 'mooring') && ~isempty(output.mooring)
        Tmoor = output.mooring.forceTotal(:,3);
    else
        Tmoor = zeros(size(t));                     % linear mooring may not log force
    end

    T = table(t, z1, z2, rel, vrel, Fpto, Ppto, Tmoor);

    rawDir = fullfile('data','raw');
    if ~exist(rawDir, 'dir'), mkdir(rawDir); end

    fn = fullfile(rawDir, sprintf('run_%05d__%s_sev%02d_Hs%03d_Tp%03d.csv', ...
                  runID, faultName, round(severity*100), round(Hs*100), round(Tp*10)));
    writetable(T, fn);

    % Append a row to the manifest.
    manifest = fullfile(rawDir, 'manifest.csv');
    newRow = table(runID, string(faultName), label, severity, Hs, Tp, string(fn), ...
        'VariableNames', {'runID','fault','label','severity','Hs','Tp','file'});
    if exist(manifest, 'file')
        writetable(newRow, manifest, 'WriteMode', 'append');
    else
        writetable(newRow, manifest);
    end

    fprintf('[export] wrote %s (%d rows)\n', fn, height(T));
end
