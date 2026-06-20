%% generate_degradation_run.m — fine PTO degradation sweep under IDENTICAL waves
% Damping ramps 1.0 -> 0.2 in 0.05 steps (17 health levels), same real spectrum + same phase
% seed every run, so the ONLY thing changing is PTO health. Stitched in Python (tools/run_alert.py
% / run_leadtime.py with the degradation folder) -> smooth degradation timeline for authentic
% lead-time / time-to-failure. Output kept SEPARATE from the main classification dataset.
%
% PREREQ on MATLAB Online: RM3MooringMatrix.slx in examples/RM3, rm3.h5 via bemio,
%                          real_spectra.mat uploaded to /MATLAB Drive.
clear; clc;
wecSimRoot = '/MATLAB Drive/WEC-Sim';
rm3Dir     = fullfile(wecSimRoot,'examples','RM3');
outDir     = '/MATLAB Drive/degradation/raw';
specFile   = '/MATLAB Drive/real_spectra.mat';
SPEC_IDX   = 5;                                       % moderate sea state (~Hs 3 m)
ptoNom = 1200000;  moorNom = 1e5;  endTime = 300;
levels = 1.0:-0.05:0.2;                               % 17 health levels

SP = load(specFile);
addpath(genpath(wecSimRoot));
if ~exist(outDir,'dir'); mkdir(outDir); end
delete(fullfile(outDir,'run_*.csv'));
writematrix([SP.freqs(:), SP.S_all(SPEC_IDX,:)'], fullfile(rm3Dir,'spectrum_tmp.txt'), 'Delimiter','tab');
Hs = SP.Hs(SPEC_IDX);  Tp = SP.Tp(SPEC_IDX);
origInput = fullfile(rm3Dir,'wecSimInputFile.m');
backup    = fullfile(rm3Dir,'wecSimInputFile_STOCK_BACKUP.m');
if ~exist(backup,'file'); copyfile(origInput, backup); end
cleanupObj = onCleanup(@() restoreInput(backup, origInput));
cd(rm3Dir);  runID = 0;  rows = {};
for lvl = levels
    runID = runID + 1;  ptoDamp = ptoNom*lvl;
    fprintf('\n=== level %d/%d | damping x%.2f (%.0f) ===\n', runID, numel(levels), lvl, ptoDamp);
    writeInputFile(origInput, ptoDamp, moorNom, endTime);
    try
        wecSim;
        fn = sprintf('run_%05d__degradation_sev%02d_Hs%03d_Tp%03d.csv', ...
                     runID, round(lvl*100), round(Hs*100), round(Tp*10));
        exportRun(output, fullfile(outDir,fn));
        rows(end+1,:) = {runID, lvl, Hs, Tp, fn};
        fprintf('  OK -> %s\n', fn);
    catch ME
        fprintf(2,'  FAILED: %s\n', ME.message);
    end
end
writetable(cell2table(rows,'VariableNames',{'runID','health','Hs','Tp','file'}), fullfile(outDir,'manifest.csv'));
fprintf('\nDONE: %d health levels in %s\n', runID, outDir);

function writeInputFile(path, ptoDamp, moorK, endTime)
    L = {
      'simu = simulationClass();'
      'simu.simMechanicsFile = ''RM3MooringMatrix.slx'';'
      'simu.mode = ''normal'';  simu.explorer = ''off'';'
      'simu.startTime = 0;  simu.rampTime = 100;'
      sprintf('simu.endTime = %g;', endTime)
      'simu.solver = ''ode4'';  simu.dt = 0.1;'
      'waves = waveClass(''spectrumImport'');'
      'waves.spectrumFile = ''spectrum_tmp.txt'';'
      'waves.phaseSeed = 1;'
      'body(1) = bodyClass(''hydroData/rm3.h5'');  body(1).geometryFile = ''geometry/float.stl'';'
      'body(1).mass = ''equilibrium'';  body(1).inertia = [20907301 21306090.66 37085481.11];'
      'body(2) = bodyClass(''hydroData/rm3.h5'');  body(2).geometryFile = ''geometry/plate.stl'';'
      'body(2).mass = ''equilibrium'';  body(2).inertia = [94419614.57 94407091.24 28542224.82];'
      'body(2).initial.displacement = [0 0 -0.21];'
      'constraint(1) = constraintClass(''Constraint1'');  constraint(1).location = [0 0 0];'
      'pto(1) = ptoClass(''PTO1'');  pto(1).stiffness = 0;'
      sprintf('pto(1).damping = %g;', ptoDamp)
      'pto(1).location = [0 0 0];'
      'mooring(1) = mooringClass(''mooring'');'
      'mooring(1).matrix.stiffness = zeros(6,6);'
      sprintf('mooring(1).matrix.stiffness(1,1) = %g;', moorK)
      'mooring(1).matrix.damping = zeros(6,6);  mooring(1).matrix.preTension = zeros(1,6);'
    };
    fid = fopen(path,'w'); fprintf(fid,'%s\n', L{:}); fclose(fid);
end
function exportRun(output, filepath)
    t=output.bodies(1).time; z1=output.bodies(1).position(:,3); z2=output.bodies(2).position(:,3);
    rel=z1-z2; vrel=output.ptos(1).velocity(:,3);
    Fpto=output.ptos(1).forceInternalMechanics(:,3); Ppto=output.ptos(1).powerInternalMechanics(:,3);
    x1=output.bodies(1).position(:,1); x2=output.bodies(2).position(:,1);
    try; Tmoor=output.mooring(1).forceMooring(:,1); catch; Tmoor=zeros(numel(t),1); end
    writetable(table(t,z1,z2,rel,vrel,Fpto,Ppto,x1,x2,Tmoor), filepath);
end
function restoreInput(backup, origInput)
    if exist(backup,'file'); copyfile(backup, origInput); end
end
