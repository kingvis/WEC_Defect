%% run_batch.m — generate the whole RM3 fault dataset (MCR-style batch driver)
% Run from the repo root on MATLAB Online, with WEC-Sim on the path and the RM3
% example assets (RM3.slx, hydroData/, geometry/) available in this folder.
%
% Workflow per run:
%   set Hs,Tp,faultName,sev in the workspace -> run wecSimInputFile_template.m
%   -> wecSim -> export_timeseries to data/raw/.

clear; clc;

%% Fault list + severities (mirror config/faults.yaml)
faults = {'healthy','pto_damping_loss','mooring_stiffness_loss','pto_plus_mooring'};
labels = containers.Map( ...
    {'healthy','pto_damping_loss','mooring_stiffness_loss','pto_plus_mooring'}, ...
    {0, 1, 2, 4});

severity = struct();
severity.healthy                = 1.0;
severity.pto_damping_loss       = [0.8 0.6 0.4 0.2];
severity.mooring_stiffness_loss = [0.75 0.5 0.25];
severity.pto_plus_mooring       = 0.5;          % representative compound case

%% Sea-state grid
% For first runs, a small hardcoded grid keeps batch time manageable.
% Later: parse config/sea_states.yaml (e.g. with a YAML reader) for the full matrix.
seaStates = struct('Hs', {0.75, 1.75, 3.0, 0.75, 1.75, 3.0}, ...
                   'Tp', {6.0,  9.0,  12.0, 12.0, 6.0, 9.0});

%% Batch loop
runID = 0;
for f = 1:numel(faults)
  fname = faults{f};
  for sev = severity.(fname)
    for ss = 1:numel(seaStates)
      runID = runID + 1;
      Hs = seaStates(ss).Hs;
      Tp = seaStates(ss).Tp;

      fprintf('\n=== run %d | %s sev=%.2f Hs=%.2f Tp=%.2f ===\n', runID, fname, sev, Hs, Tp);

      try
        run('wecSimInputFile_template.m');   % builds RM3 objects + injects fault
        wecSim;                              % run the simulation
        export_timeseries(output, fname, labels(fname), sev, Hs, Tp, runID);
      catch ME
        fprintf(2, '  RUN %d FAILED: %s\n', runID, ME.message);
      end
    end
  end
end

fprintf('\nBatch complete: %d runs attempted. CSVs in data/raw/.\n', runID);
