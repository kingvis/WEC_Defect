%% generate_unified_dataset.m — UNIFIED 4-class RM3 fault dataset (moored model)
% All runs use the moored model RM3MooringMatrix.slx so every class shares the same
% baseline (a real WEC is always moored). Faults are deviations from nominal:
%   healthy                : nominal PTO damping + nominal mooring stiffness
%   pto_damping_loss       : PTO damping reduced
%   mooring_stiffness_loss : mooring surge stiffness reduced
%   pto_plus_mooring       : BOTH reduced (compound)
%
% Channels exported (9 + time): heave side (PTO) + surge side (mooring):
%   t, z1, z2, rel, vrel, Fpto, Ppto, x1, x2, Tmoor
%
% PREREQ: RM3MooringMatrix.slx copied into examples/RM3/ (see Phase-1 Step 1),
%         and rm3.h5 already generated via bemio.
% Usage:  edit('/MATLAB Drive/generate_unified_dataset.m'); paste; save;
%         run('/MATLAB Drive/generate_unified_dataset.m')
clear; clc;

%% ---------- CONFIG ----------
wecSimRoot = '/MATLAB Drive/WEC-Sim';
rm3Dir     = fullfile(wecSimRoot,'examples','RM3');
outDir     = '/MATLAB Drive/wec_data/raw';

ptoNom  = 1200000;        % nominal PTO damping [N/(m/s)]
moorNom = 1e5;            % nominal mooring surge stiffness [N/m]
endTime = 400;

% fault name | label | severity multipliers
specs = {
  'healthy',                0, 1.0;
  'pto_damping_loss',       1, [0.8 0.6 0.4 0.2];
  'mooring_stiffness_loss', 2, [0.75 0.5 0.25];
  'pto_plus_mooring',       4, 0.5;
};
% REAL sea states from NDBC buoys 46022/46050/46026 (2020-2022); see config/sea_states.yaml
seaStates = [1.19 7.7; 1.19 10.8; 1.19 14.8;     % 3x3 grid (Hs P15/P50/P90 x Tp P15/P50/P90)
             1.92 7.7; 1.92 10.8; 1.92 14.8;
             3.59 7.7; 3.59 10.8; 3.59 14.8];
% ----------------------------

addpath(genpath(wecSimRoot));
if ~exist(outDir,'dir'); mkdir(outDir); end
% Clear any previous (incompatible) CSVs so the dataset is consistent.
delete(fullfile(outDir,'run_*.csv'));
if isfile(fullfile(outDir,'manifest.csv')); delete(fullfile(outDir,'manifest.csv')); end

origInput = fullfile(rm3Dir,'wecSimInputFile.m');
backup    = fullfile(rm3Dir,'wecSimInputFile_STOCK_BACKUP.m');
if ~exist(backup,'file'); copyfile(origInput, backup); end
cleanupObj = onCleanup(@() restoreInput(backup, origInput));

cd(rm3Dir);
runID = 0;  rows = {};
for si = 1:size(specs,1)
    fname = specs{si,1};  label = specs{si,2};  sevs = specs{si,3};
    for sev = sevs
        ptoDamp = ptoNom;  moorK = moorNom;
        if any(strcmp(fname,{'pto_damping_loss','pto_plus_mooring'}));       ptoDamp = ptoNom*sev;  end
        if any(strcmp(fname,{'mooring_stiffness_loss','pto_plus_mooring'})); moorK   = moorNom*sev; end
        for ss = 1:size(seaStates,1)
            Hs = seaStates(ss,1);  Tp = seaStates(ss,2);  runID = runID + 1;
            fprintf('\n=== run %d | %s sev=%.2f Hs=%.2f Tp=%.2f ptoDamp=%.0f moorK=%.0f ===\n', ...
                    runID, fname, sev, Hs, Tp, ptoDamp, moorK);
            writeInputFile(origInput, Hs, Tp, ptoDamp, moorK, endTime);
            try
                wecSim;
                fn = sprintf('run_%05d__%s_sev%02d_Hs%03d_Tp%03d.csv', ...
                             runID, fname, round(sev*100), round(Hs*100), round(Tp*10));
                exportRun(output, fullfile(outDir,fn));
                rows(end+1,:) = {runID, fname, label, sev, Hs, Tp, fn};
                fprintf('  OK -> %s\n', fn);
            catch ME
                fprintf(2,'  FAILED: %s\n', ME.message);
            end
        end
    end
end
M = cell2table(rows, 'VariableNames', {'runID','fault','label','severity','Hs','Tp','file'});
writetable(M, fullfile(outDir,'manifest.csv'));
fprintf('\nDONE: %d runs. CSVs + manifest in %s\n', runID, outDir);

%% ---------- helpers ----------
function writeInputFile(path, Hs, Tp, ptoDamp, moorK, endTime)
    L = {
      '%% Auto-generated WEC-Sim input file (unified fault batch, moored)'
      'simu = simulationClass();'
      'simu.simMechanicsFile = ''RM3MooringMatrix.slx'';'
      'simu.mode = ''normal'';'
      'simu.explorer = ''off'';'
      'simu.startTime = 0;'
      'simu.rampTime = 100;'
      sprintf('simu.endTime = %g;', endTime)
      'simu.solver = ''ode4'';'
      'simu.dt = 0.1;'
      'waves = waveClass(''irregular'');'
      sprintf('waves.height = %g;', Hs)
      sprintf('waves.period = %g;', Tp)
      'waves.spectrumType = ''JS'';'
      'waves.phaseSeed = 1;'
      'body(1) = bodyClass(''hydroData/rm3.h5'');'
      'body(1).geometryFile = ''geometry/float.stl'';'
      'body(1).mass = ''equilibrium'';'
      'body(1).inertia = [20907301 21306090.66 37085481.11];'
      'body(2) = bodyClass(''hydroData/rm3.h5'');'
      'body(2).geometryFile = ''geometry/plate.stl'';'
      'body(2).mass = ''equilibrium'';'
      'body(2).inertia = [94419614.57 94407091.24 28542224.82];'
      'body(2).initial.displacement = [0 0 -0.21];'
      'constraint(1) = constraintClass(''Constraint1'');'
      'constraint(1).location = [0 0 0];'
      'pto(1) = ptoClass(''PTO1'');'
      'pto(1).stiffness = 0;'
      sprintf('pto(1).damping = %g;', ptoDamp)
      'pto(1).location = [0 0 0];'
      'mooring(1) = mooringClass(''mooring'');'
      'mooring(1).matrix.stiffness = zeros(6,6);'
      sprintf('mooring(1).matrix.stiffness(1,1) = %g;', moorK)
      'mooring(1).matrix.damping = zeros(6,6);'
      'mooring(1).matrix.preTension = zeros(1,6);'
    };
    fid = fopen(path,'w'); fprintf(fid,'%s\n', L{:}); fclose(fid);
end

function exportRun(output, filepath)
    t    = output.bodies(1).time;
    z1   = output.bodies(1).position(:,3);          % float heave
    z2   = output.bodies(2).position(:,3);          % spar heave
    rel  = z1 - z2;                                 % relative heave (PTO)
    vrel = output.ptos(1).velocity(:,3);
    Fpto = output.ptos(1).forceInternalMechanics(:,3);
    Ppto = output.ptos(1).powerInternalMechanics(:,3);
    x1   = output.bodies(1).position(:,1);          % float surge (mooring drift)
    x2   = output.bodies(2).position(:,1);          % spar surge
    try
        Tmoor = output.mooring(1).forceMooring(:,1);  % mooring surge force
    catch
        Tmoor = zeros(numel(t),1);                    % (output is an object; isfield fails on it)
    end
    T = table(t,z1,z2,rel,vrel,Fpto,Ppto,x1,x2,Tmoor);
    writetable(T, filepath);
end

function restoreInput(backup, origInput)
    if exist(backup,'file'); copyfile(backup, origInput); end
end
