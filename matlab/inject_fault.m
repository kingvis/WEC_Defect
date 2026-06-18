function [pto, mooring] = inject_fault(pto, mooring, faultName, severity)
% INJECT_FAULT  Apply one fault config to the WEC-Sim input objects.
%
%   [pto, mooring] = inject_fault(pto, mooring, faultName, severity)
%
% Called inside wecSimInputFile_template.m AFTER the healthy objects are defined
% and BEFORE wecSim runs. The two core faults are implemented explicitly; the
% architecture (faults defined in config/faults.yaml) supports adding more without
% changing the ML side.
%
% Inputs:
%   pto, mooring : WEC-Sim objects (passed in/out so changes persist)
%   faultName    : 'healthy' | 'pto_damping_loss' | 'mooring_stiffness_loss' | 'pto_plus_mooring'
%   severity     : multiplier in (0,1]; 1.0 = healthy. Meaning depends on the fault.

    switch faultName
        case 'healthy'
            % no change

        case 'pto_damping_loss'
            % Reduce PTO linear damping (seal wear / fluid loss).
            pto(1).c = pto(1).c * severity;

        case 'mooring_stiffness_loss'
            % Linear mooring matrix path: scale heave stiffness term.
            % If using MoorDyn instead, scale line stiffness in the MoorDyn input
            % file rather than here (documented upgrade path, see guide §5.4).
            mooring(1).matrix(3,3) = mooring(1).matrix(3,3) * severity;

        case 'pto_plus_mooring'
            % Compound fault: both subsystems degraded simultaneously.
            pto(1).c = pto(1).c * severity;
            mooring(1).matrix(3,3) = mooring(1).matrix(3,3) * severity;

        otherwise
            error('inject_fault:unknownFault', 'Unknown fault: %s', faultName);
    end
end
