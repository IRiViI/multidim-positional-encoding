import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        return emb[None, :, :orig_ch].repeat(batch_size, 1, 1)


class PositionalEncodingPermute1D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 2, 1)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


class DiagonalPositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(DiagonalPositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Diagonal Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        diagonal_len = x + y - 1
        pos = torch.arange(diagonal_len, device=tensor.device).type(self.inv_freq.type())

        # Diagonal l
        sin_inp_l = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb_l = torch.cat((sin_inp_l.sin(), sin_inp_l.cos()), dim=-1)
        emb_rep_l = torch.zeros((x, diagonal_len, self.channels), device=tensor.device).type(
            tensor.type()
        )
        emb_rep_l[:,:,:]=emb_l

        # Diagonal r
        sin_inp_r = torch.einsum("i,j->ij", pos-x+1, self.inv_freq)
        emb_r = torch.cat((sin_inp_r.sin(), sin_inp_r.cos()), dim=-1)
        emb_rep_r = torch.zeros((x, diagonal_len, self.channels), device=tensor.device).type(
            tensor.type()
        )
        emb_rep_r[:,:,:]=emb_r
        
        # Selection masks
        d = x + y -1
        x_s = torch.arange(x)[:,None] * torch.ones(d)
        y_s = torch.ones(x)[:,None] * torch.arange(d)
        
        zr_s = x_s+y_s
        fr_s = (zr_s >= x-1) & (zr_s < d)

        zl_s = y_s - x_s
        fl_s = (zl_s >= 0) & (zl_s <= d-x)

        # Selection
        emb_s = torch.zeros((x, y, 2*self.channels), device=tensor.device).type(
            tensor.type()
        )
        emb_s[:,:,:self.channels] = emb_rep_r[fr_s].view(x,y,self.channels)
        emb_s[:,:,self.channels:] = emb_rep_l[fl_s].view(x,y,self.channels)

        return emb_s[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)


class DiagonalPositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(DiagonalPositionalEncodingPermute2D, self).__init__()
        self.penc = DiagonalPositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = (
            torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        return emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)


class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels


class FixEncoding(nn.Module):
    """
    :param pos_encoder: instance of PositionalEncoding1D, PositionalEncoding2D or PositionalEncoding3D
    :param shape: shape of input, excluding batch and embedding size
    Example:
    p_enc_2d = FixEncoding(PositionalEncoding2D(32), (x, y)) # for where x and y are the dimensions of your image
    inputs = torch.randn(64, 128, 128, 32) # where x and y are 128, and 64 is the batch size
    p_enc_2d(inputs)
    """

    def __init__(self, pos_encoder, shape):
        super(FixEncoding, self).__init__()
        self.shape = shape
        self.dim = len(shape)
        self.pos_encoder = pos_encoder
        self.pos_encoding = pos_encoder(
            torch.ones(1, *shape, self.pos_encoder.org_channels)
        )
        self.batch_size = 0

    def forward(self, tensor):
        if self.batch_size != tensor.shape[0]:
            self.repeated_pos_encoding = self.pos_encoding.to(tensor.device).repeat(
                tensor.shape[0], *(self.dim + 1) * [1]
            )
            self.batch_size = tensor.shape[0]
        return self.repeated_pos_encoding

class DynamicEncoding(nn.Module):
    """
    :param pos_encoder: instance of PositionalEncoding1D, PositionalEncoding2D or PositionalEncoding3D
    :param median_batch_size: The most common batch size
    Example:
    p_enc_2d = DynamicEncoding(PositionalEncoding2D(32), 16)
    inputs = torch.randn(16, 128, 128, 32)
    p_enc_2d(inputs)
    """

    def __init__(self, pos_encoder, median_batch_size:int):
        super(DynamicEncoding, self).__init__()
        self.median_batch_size:int = median_batch_size
        self.pos_encoder = pos_encoder
        self.pos_encodings = {}
        self.batch_pos_encodings = {}
        self.batch_size = 0

    def forward(self, tensor):
        batch_size = tensor.shape[0]
        shape = tensor.shape[1:-1]
        dim = len(shape)
        # Check if the pos encoding has already been used with the corresponding batch size
        if shape not in self.batch_pos_encodings:
            if shape not in self.pos_encodings:
                pos_encoding = self.pos_encoder(
                    torch.ones(1, *shape, self.pos_encoder.org_channels)
                ).to(tensor.device)
                self.pos_encodings[shape] = pos_encoding
            else:
                pos_encoding = self.pos_encodings[shape]
            batch_pos_encoding = pos_encoding.to(tensor.device).repeat(
                batch_size, *(dim + 1) * [1]
            )
            if batch_size == self.median_batch_size:
                self.batch_pos_encodings[shape] = batch_pos_encoding
        else:
            batch_pos_encoding = self.pos_encodings[shape]
        return batch_pos_encoding