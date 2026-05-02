%% Silver-ion battery ECM simulation
% Sign convention:
% I > 0  -> discharge
% I < 0  -> charge

clear; clc;

%% Parameters
p.Q_Ah = 2.0;
p.R0_Ohm = 0.035;
p.R1_Ohm = 0.020;
p.C1_F = 2200.0;
p.eta_charge = 0.995;
p.eta_discharge = 1.0;
p.V_min = 2.8;
p.V_max = 4.2;

%% Simulation setup
dt_s = 1.0;
t_end_s = 9000.0;
t_s = 0:dt_s:t_end_s;
n = numel(t_s);

soc0 = 0.95;
vp0_V = 0.0;

I_A = current_profile(t_s);

soc = zeros(1, n);
vp_V = zeros(1, n);

soc(1) = min(max(soc0, 0.0), 1.0);
vp_V(1) = vp0_V;

Q_As = p.Q_Ah * 3600.0;

%% Time stepping (Forward Euler)
for k = 1:(n - 1)
    Ik = I_A(k);

    if Ik >= 0
        eta = p.eta_discharge;
    else
        eta = p.eta_charge;
    end

    dsoc_dt = -(eta * Ik) / Q_As;
    dvp_dt = -(vp_V(k) / (p.R1_Ohm * p.C1_F)) + (Ik / p.C1_F);

    soc(k + 1) = min(max(soc(k) + dt_s * dsoc_dt, 0.0), 1.0);
    vp_V(k + 1) = vp_V(k) + dt_s * dvp_dt;
end

OCV_V = ocv_from_soc(soc);
Vt_V = OCV_V - I_A .* p.R0_Ohm - vp_V;

%% Plot
figure('Color', 'w', 'Position', [100 80 900 760]);

tiledlayout(4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(t_s, I_A, 'LineWidth', 1.5);
ylabel('Current (A)');
grid on;

nexttile;
plot(t_s, soc, 'LineWidth', 1.5);
ylabel('SOC');
ylim([-0.02 1.02]);
grid on;

nexttile;
plot(t_s, vp_V, 'LineWidth', 1.5);
ylabel('V_p (V)');
grid on;

nexttile;
plot(t_s, Vt_V, 'LineWidth', 1.7);
hold on;
yline(p.V_min, '--', 'V_min');
yline(p.V_max, '--', 'V_max');
ylabel('Voltage (V)');
xlabel('Time (s)');
grid on;
legend({'Terminal voltage', 'V_{min}', 'V_{max}'}, 'Location', 'best');

sgtitle('Silver-Ion Battery ECM Simulation');

%% Console summary
fprintf('Simulation complete\n');
fprintf('Final SOC: %.4f\n', soc(end));
fprintf('Final terminal voltage: %.4f V\n', Vt_V(end));

%% Output struct
sim_out = struct();
sim_out.t_s = t_s;
sim_out.I_A = I_A;
sim_out.soc = soc;
sim_out.vp_V = vp_V;
sim_out.OCV_V = OCV_V;
sim_out.Vt_V = Vt_V;

%% Local functions
function I = current_profile(t)
    I = zeros(size(t));
    I((t >= 600.0) & (t < 4200.0)) = 1.0;
    I((t >= 4800.0) & (t < 7800.0)) = -0.8;
end

function ocv = ocv_from_soc(soc)
    eps_val = 1e-6;
    s = min(max(soc, eps_val), 1.0 - eps_val);

    a0 = 3.10;
    a1 = 1.00;
    a2 = -0.22;
    a3 = 0.05;
    a4 = -0.04;

    ocv = a0 + a1 .* s + a2 .* (s .^ 2) + a3 .* log(s) + a4 .* log(1.0 - s);
end
