%% run_batch_simple.m — Simplified fault generation for RM3 on MATLAB Online
% This script generates healthy + faulty RM3 simulations and exports CSVs.
% 
% Usage (in MATLAB Online):
%   1. cd ~/WEC_Defect
%   2. addpath(genpath('config'))
%   3. run('matlab/run_batch_simple.m')
%
% It will create data/raw/*.csv files that you can download.

clear; clc;

%% Setup paths
wec_sim_path = '/home/matlab/WEC-Sim';
rm3_example = [wec_sim_path '/examples/RM3'];
addpath(wec_sim_path);
addpath(rm3_example);

% Verify WEC-Sim is available
try
    wecSimVersion;
    disp('✓ WEC-Sim found and loaded');
catch
    error('WEC-Sim not on path. Make sure it is cloned in MATLAB Online.');
end

%% Create output directory
data_raw = 'data/raw';
if ~exist(data_raw, 'dir')
    mkdir(data_raw);
    fprintf('✓ Created %s\n', data_raw);
end

%% Fault configuration
faults = {'healthy', 'pto_damping_loss', 'mooring_stiffness_loss', 'pto_plus_mooring'};
fault_labels = containers.Map( ...
    faults, ...
    {0, 1, 2, 4});

% Severity levels per fault
severities = containers.Map();
severities('healthy') = 1.0;
severities('pto_damping_loss') = [0.8, 0.6, 0.4, 0.2];
severities('mooring_stiffness_loss') = [0.75, 0.5, 0.25];
severities('pto_plus_mooring') = 0.5;

%% Small sea-state grid for testing (expand later)
sea_states = struct('Hs', {0.75, 1.75, 3.0}, ...
                    'Tp', {6.0, 9.0, 12.0});

%% Batch loop
runID = 0;
manifest = {};

for f = 1:length(faults)
    fname = faults{f};
    sev_list = severities(fname);
    
    % Make sure it's a row vector
    if ~isrow(sev_list)
        sev_list = sev_list(:)';
    end
    
    for sev = sev_list
        for ss = 1:length(sea_states)
            runID = runID + 1;
            Hs = sea_states(ss).Hs;
            Tp = sea_states(ss).Tp;
            label = fault_labels(fname);
            
            fprintf('\n=== RUN %03d | %s (sev=%.2f) | Hs=%.2f Tp=%.2f ===\n', ...
                runID, fname, sev, Hs, Tp);
            
            try
                % Generate CSV filename
                sev_int = round(sev * 100);
                hs_int = round(Hs * 100);
                tp_int = round(Tp * 10);
                filename = sprintf('run_%05d__%s_sev%d_Hs%d_Tp%03d.csv', ...
                    runID, fname, sev_int, hs_int, tp_int);
                filepath = fullfile(data_raw, filename);
                
                % Copy RM3 example input file and modify it
                input_file = [rm3_example '/wecSimInputFile.m'];
                local_input = 'wecSimInputFile_modified.m';
                
                % Read original
                fid = fopen(input_file, 'r');
                content = fread(fid, '*char')';
                fclose(fid);
                
                % Inject fault parameters at the end (before simu.caseDir)
                injection = sprintf('\n%% FAULT INJECTION\nHs_fault = %.2f;\nTp_fault = %.2f;\nfault_name = ''%s'';\nfault_sev = %.2f;\n', ...
                    Hs, Tp, fname, sev);
                content = [content, injection];
                
                % Write modified file
                fid = fopen(local_input, 'w');
                fwrite(fid, content);
                fclose(fid);
                
                % Change to RM3 example directory (WEC-Sim expects to be there)
                original_dir = pwd();
                cd(rm3_example);
                
                % Run the simulation
                run(local_input);  % Loads RM3 model with modified Hs/Tp
                wecSim;            % Runs simulation
                
                % Export to CSV
                export_simple_csv(output, filepath, label);
                
                % Return to original directory
                cd(original_dir);
                
                % Log to manifest
                manifest{runID} = struct(...
                    'runID', runID, ...
                    'fault', fname, ...
                    'label', label, ...
                    'severity', sev, ...
                    'Hs', Hs, ...
                    'Tp', Tp, ...
                    'filename', filename);
                
                fprintf('✓ Exported %s\n', filename);
                
            catch ME
                fprintf(2, '✗ RUN %d FAILED: %s\n', runID, ME.message);
            end
        end
    end
end

%% Write manifest
manifest_file = fullfile(data_raw, 'manifest.csv');
write_manifest_csv(manifest, manifest_file);
fprintf('\n✓ Manifest written to %s\n', manifest_file);
fprintf('✓ Batch complete! %d runs generated.\n', runID);

%% Helper: Export output to simple CSV
function export_simple_csv(output, filepath, label)
    % Extract time-series from WEC-Sim output object
    t = output.time;
    z1 = output.bodies(1).position(:, 3);  % float heave
    z2 = output.bodies(2).position(:, 3);  % spar heave
    rel = z1 - z2;                          % relative heave
    
    % Approximate velocities (numerical derivative)
    vrel = [0; diff(rel) ./ diff(t)];
    
    % PTO force and power
    Fpto = output.pto(1).force(:, 3);
    Ppto = output.pto(1).power;
    
    % Mooring tension (if available, else zeros)
    if isfield(output, 'mooring') && ~isempty(output.mooring)
        Tmoor = output.mooring(1).forceTotal(:, 3);
    else
        Tmoor = zeros(length(t), 1);
    end
    
    % Assemble table and write
    T = table(t, z1, z2, rel, vrel, Fpto, Ppto, Tmoor);
    writetable(T, filepath);
end

%% Helper: Write manifest CSV
function write_manifest_csv(manifest, filepath)
    if isempty(manifest)
        warning('Manifest is empty.');
        return;
    end
    
    % Convert cell array of structs to table
    data = [];
    for i = 1:length(manifest)
        if ~isempty(manifest{i})
            data = [data; manifest{i}];
        end
    end
    
    if isempty(data)
        warning('No valid manifest entries.');
        return;
    end
    
    T = struct2table(data);
    writetable(T, filepath);
end
