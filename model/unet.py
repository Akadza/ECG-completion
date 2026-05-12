import torch
import torch.nn as nn

class LeadInteractionBlock(nn.Module):
    def __init__(self, n_leads=12, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(n_leads, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(d_model, n_leads)
        self.norm1 = nn.LayerNorm(n_leads)
        self.norm2 = nn.LayerNorm(n_leads)
        self.ff = nn.Sequential(nn.Linear(n_leads, n_leads*2), nn.GELU(), nn.Linear(n_leads*2, n_leads))

    def forward(self, x):
        B, L, T = x.shape
        xt = x.permute(0, 2, 1)
        x_proj = self.in_proj(xt)
        attn_out, _ = self.attn(x_proj, x_proj, x_proj)
        xt = self.norm1(xt + self.out_proj(attn_out))
        return self.norm2(xt + self.ff(xt)).permute(0, 2, 1)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))) + self.skip(x))

class ECGRecoverUNetV2(nn.Module):
    def __init__(self, n_leads=12, base_channels=64):
        super().__init__()
        in_ch = n_leads * 2
        self.enc1 = ConvBlock(in_ch, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels*2)
        self.enc3 = ConvBlock(base_channels*2, base_channels*4)
        self.enc4 = ConvBlock(base_channels*4, base_channels*8)
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels*8, base_channels*8, dilation=2),
            ConvBlock(base_channels*8, base_channels*8, dilation=4),
            ConvBlock(base_channels*8, base_channels*8, dilation=8)
        )
        self.lead_proj_down = nn.Conv1d(base_channels*8, n_leads, 1)
        self.lead_interact  = LeadInteractionBlock(n_leads, 64, 4)
        self.lead_proj_up   = nn.Conv1d(n_leads, base_channels*8, 1)
        self.lead_norm      = nn.BatchNorm1d(base_channels*8)
        self.up4 = nn.ConvTranspose1d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec4 = ConvBlock(base_channels*8, base_channels*4)
        self.up3 = nn.ConvTranspose1d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec3 = ConvBlock(base_channels*4, base_channels*2)
        self.up2 = nn.ConvTranspose1d(base_channels*2, base_channels, 2, stride=2)
        self.dec2 = ConvBlock(base_channels*2, base_channels)
        self.final = nn.Sequential(nn.Conv1d(base_channels, base_channels, 3, padding=1), nn.GELU(), nn.Conv1d(base_channels, n_leads, 1))

    def forward(self, partial, mask):
        x = torch.cat([partial, mask], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(e4)
        b = self.lead_norm(self.lead_proj_up(self.lead_interact(self.lead_proj_down(b))) + b)
        d4 = self.dec4(torch.cat([self.up4(b), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        return partial + (1 - mask) * self.final(d2)