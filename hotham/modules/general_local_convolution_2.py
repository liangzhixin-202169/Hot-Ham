"""Optimized TO2 for LCAO version 1, including optional Triton kernels.

Retains the original TO2 parameters and persistent state. Fixed CUDA FP32
geometry uses fused natural-parity or general-layout kernels; other inputs use
Torch. Edge count B is dynamic and is excluded from Triton specialization.
Fused backward supports first derivatives; use backend="torch" for higher ones.
"""
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable

from .general_local_convolution import TO2Linear

try:
    import triton as tr
    import triton.language as tl
except ImportError:
    tr = tl = None

_TRITON_AVAILABLE = tr is not None

# Representation layouts and matrix products.

class GeneralLayout(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.groups = []
        self.sizes = [0, 0]
        self.dims = [op.irreps_in.dim, op.irreps_out.dim]
        self.lmax = [op.irreps_in.lmax, op.irreps_out.lmax]
        self.w = op.weight_numel
        device = op.to2_m_fcs[0].weight.device
        sides = []
        for irs in (op.irreps_in, op.irreps_out):
            channels = []
            for (mul, ir), span in zip(irs, irs.slices()):
                channels.extend((ir.l, ir.p*(-1)**ir.l, ir.t_p,
                                 span.start+c*(2*ir.l+1)) for c in range(mul))
            sides.append(channels)
        layouts = [[[ -1, 1, 0, 1, -1, -1] for _ in range(dim)] for dim in self.dims]
        gate = [None]*self.w
        bias_positions = [None]*op.to2_m_fcs[0].bias.numel()
        bias_map = {int(j):i for i,j in enumerate(op.to2_m_fcs[0].bias_idx.tolist())}
        factors, weight_meta = [], []
        self.direct = True
        for m, fc in enumerate(op.to2_m_fcs):
            cin = [ch for ch in sides[0] if ch[0]>=m]
            cout = [ch for ch in sides[1] if ch[0]>=m]
            ni, no = len(cin), len(cout)
            assert (ni,no)==(fc.c_in,fc.c_out)
            count = 1 if m==0 else 2
            self.groups.append((m, ni, no, self.sizes[0], self.sizes[1], len(factors)))
            # Packed matrices always have [Cout,Cin] orientation.
            factor = torch.tensor([[0 if ti!=to or (m==0 and qi!=qo) else
                                    (-1 if m and qi==1 and qo==-1 else 1)
                                    for li,qi,ti,bi in cin] for lo,qo,to,bo in cout], dtype=torch.int8)
            if m==0:
                assert torch.equal(factor.T==0,fc.weight_mask.cpu())
            else:
                expected = torch.tensor([[0 if ti!=to else 1 if qi==qo else 2
                                         for lo,qo,to,bo in cout] for li,qi,ti,bi in cin])
                assert torch.equal(expected,fc.cg_idx.cpu())
                assert torch.equal(fc.cgs.cpu(),torch.tensor([[[0.,0.],[0.,0.]],[[1.,0.],[0.,1.]],[[0.,1.],[-1.,0.]]]))
            self.direct &= bool((factor==1).all())
            weight_meta.append((ni,no,len(factors)))
            factors.extend(factor.flatten().tolist())
            for side, channels in enumerate((cin,cout)):
                width = len(channels)
                for col,(l,q,t,base) in enumerate(channels):
                    gidx = op.weights_slices[m]+col if side==1 else -1
                    bi = bias_map.get(col,-1) if side==1 and m==0 else -1
                    for s,j in enumerate((l,) if m==0 else (l-m,l+m)):
                        canonical_s = 1-s if m and q==-1 else s
                        sign = -1 if m and q==-1 and s==1 else 1
                        layouts[side][base+j] = [self.sizes[side],count*width,canonical_s*width+col,sign,gidx,bi]
                    if side==1:
                        gate[gidx] = [self.sizes[side],width,count,col,bi]
                        if bi>=0: bias_positions[bi] = [self.sizes[side],width,col]
                self.sizes[side] += count*width
        for side,channels in enumerate(sides):
            bases = []
            counts, starts = [], []
            for l in range(self.lmax[side]+1):
                starts.append(len(bases))
                xs=[base for ll,q,t,base in channels if ll==l]
                counts.append(len(xs)); bases.extend(xs)
            setattr(self,f'counts{side}',tuple(counts))
            setattr(self,f'starts{side}',tuple(starts))
            self.register_buffer(f'degrees{side}',torch.tensor(list(zip(counts,starts)),dtype=torch.int32,device=device),persistent=False)
            self.register_buffer(f'layout{side}',torch.tensor(layouts[side],dtype=torch.int32,device=device),persistent=False)
            self.register_buffer(f'bases{side}',torch.tensor(bases,dtype=torch.int32,device=device),persistent=False)
        self.register_buffer('gate_meta',torch.tensor(gate,dtype=torch.int32,device=device),persistent=False)
        self.register_buffer('bias_meta',torch.tensor(bias_positions,dtype=torch.int32,device=device).reshape(-1,3),persistent=False)
        # If weights already have the required basis/orientation, no packed
        # weight kernel or quadratic sign/mask buffer is needed at runtime.
        self.register_buffer('factor',torch.tensor([] if self.direct else factors,dtype=torch.int8,device=device),persistent=False)
        self.register_buffer('weight_meta',torch.tensor(weight_meta,dtype=torch.int32,device=device),persistent=False)
        self.weight_size=len(factors)
        self.max_weight_size=max(ni*no for m,ni,no,xi,yi,wi in self.groups)
        self._pointers = None
        self._pointer_values = None

    def pointers(self, weights):
        values=tuple(w.data_ptr() for w in weights)
        if values!=self._pointer_values or self._pointers.device!=weights[0].device:
            self._pointers=torch.tensor(values,dtype=torch.int64,device=weights[0].device)
            self._pointer_values=values
        return self._pointers


class _GeneralLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,plan,b,*weights):
        packed_w=None if plan.direct else _general_pack_weights(plan,weights)
        y=x.new_empty(b*plan.sizes[1])
        for m,ni,no,xi,yi,wi in plan.groups:
            rows=b*(1 if m==0 else 2)
            w=(weights[m].T if m==0 else weights[m]) if plan.direct else packed_w[wi:wi+ni*no].view(no,ni)
            torch.mm(x[b*xi:b*xi+rows*ni].view(rows,ni),w.T,
                     out=y[b*yi:b*yi+rows*no].view(rows,no))
        ctx.save_for_backward(x,packed_w,*weights)
        ctx.plan,ctx.b=plan,b
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx,dy):
        x,packed_w,*weights=ctx.saved_tensors
        plan,b=ctx.plan,ctx.b
        dx=torch.empty_like(x) if ctx.needs_input_grad[0] else None
        need_w=any(ctx.needs_input_grad[3:])
        dw=x.new_empty(plan.weight_size) if need_w else None
        for m,ni,no,xi,yi,wi in plan.groups:
            rows=b*(1 if m==0 else 2)
            xx=x[b*xi:b*xi+rows*ni].view(rows,ni)
            yy=dy[b*yi:b*yi+rows*no].view(rows,no)
            w=(weights[m].T if m==0 else weights[m]) if plan.direct else packed_w[wi:wi+ni*no].view(no,ni)
            if dx is not None:
                torch.mm(yy,w,out=dx[b*xi:b*xi+rows*ni].view(rows,ni))
            if need_w:
                target=dw[wi:wi+ni*no].view(ni,no) if plan.direct and m==0 else dw[wi:wi+ni*no].view(no,ni)
                torch.mm(xx.T,yy,out=target) if plan.direct and m==0 else torch.mm(yy.T,xx,out=target)
        raw=dw if plan.direct or not need_w else _general_unpack_weight_grads(plan,dw)
        grads=[raw[wi:wi+ni*no].view_as(weights[m]) if ctx.needs_input_grad[3+m] else None
               for m,ni,no,xi,yi,wi in plan.groups]
        return dx,None,None,*grads


general_linear=_GeneralLinear.apply


class _StructuredLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias, b, c, lmax, *weights):
        y = torch.empty_like(x)
        offset = 0
        for m, w in enumerate(weights):
            width = c*(lmax-m+1)
            rows = b if m == 0 else 2*b
            size = rows*width
            xx = x[offset:offset+size].view(rows,width)
            yy = y[offset:offset+size].view(rows,width)
            torch.mm(xx, w if m == 0 else w.T, out=yy)
            if m == 0:
                yy[:, :c].add_(bias)
            offset += size
        ctx.save_for_backward(x, *weights)
        ctx.b,ctx.c,ctx.lmax = b,c,lmax
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x,*weights = ctx.saved_tensors
        b,c,lmax = ctx.b,ctx.c,ctx.lmax
        dx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
        db = None
        dweights = []
        offset = 0
        for m,w in enumerate(weights):
            width = c*(lmax-m+1)
            rows = b if m == 0 else 2*b
            size = rows*width
            xx = x[offset:offset+size].view(rows,width)
            dy = grad[offset:offset+size].view(rows,width)
            if dx is not None:
                torch.mm(dy, w.T if m == 0 else w,
                         out=dx[offset:offset+size].view(rows,width))
            dw = None
            if ctx.needs_input_grad[5+m]:
                dw = torch.mm(xx.T,dy) if m == 0 else torch.mm(dy.T,xx)
            if m == 0 and ctx.needs_input_grad[1]:
                db = dy[:,:c].sum(0)
            dweights.append(dw)
            offset += size
        return dx,db,None,None,None,*dweights


structured_linear = _StructuredLinear.apply


# Optional CUDA kernels.
if _TRITON_AVAILABLE:

    # Kept as an explicit profiling switch; formal comparisons use "auto".
    _natural_ROTATION_VARIANT = "auto"


    @tr.jit(do_not_specialize=["B"])
    def _natural_rotation(X, D, GATE, MASK, Y, B, C: tl.constexpr,
                  L: tl.constexpr, PACK: tl.constexpr, GATED: tl.constexpr,
                  BC: tl.constexpr, BJ: tl.constexpr, DEGREE: tl.constexpr = -1):
        l = tl.program_id(1) if DEGREE < 0 else tl.full((), DEGREE, tl.int32)
        row = tl.program_id(0)*BC + tl.arange(0, BC)
        b, c = row // C, row % C
        j = tl.arange(0, BJ)
        d = 2*l+1
        dim: tl.constexpr = C*(L+1)*(L+1)
        dtotal: tl.constexpr = (L+1)*(4*(L+1)*(L+1)-1)//3
        doff = l*(4*l*l-1)//3
        active = tl.load(MASK+b, b<B, other=0)
        acc = tl.full((BC, BJ), 0, tl.float32)
        # PACK: x_world @ D -> contiguous m GEMM rows.
        # UNPACK: m rows (optionally gated) @ D.T -> world irreps.
        for k in range(2*L+1):
            if PACK:
                v = tl.load(X+b*dim+C*l*l+c*d+k, (b<B)&(k<d)&active, other=0)
                dv = tl.load(D+b[:,None]*dtotal+doff+k*d+j[None,:],
                             (b[:,None]<B)&(j[None,:]<d)&(k<d)&active[:,None], other=0)
                dv = tl.where(l == 0, tl.where(j[None,:] == k, 1.0, 0.0), dv)
                acc += v[:,None]*dv
            else:
                m = tl.abs(k-l)
                s = tl.where(k>l, 1, 0)
                count = tl.where(m==0, 1, 2)
                width = C*(L-m+1)
                base = tl.where(m==0, 0, C*((2*m-1)*(L+1)-m*(m-1)))
                pos = base*B+(b*count+s)*width+(l-m)*C+c
                v = tl.load(X+pos, (b<B)&(k<d)&active, other=0)
                if GATED:
                    goff = C*m*(2*L+3-m)//2
                    g = tl.load(GATE+b*(C*(L+1)*(L+2)//2)+goff+(l-m)*C+c,
                                (b<B)&(k<d)&active, other=0)
                    v *= g
                dv = tl.load(D+b[:,None]*dtotal+doff+j[None,:]*d+k,
                             (b[:,None]<B)&(j[None,:]<d)&(k<d)&active[:,None], other=0)
                dv = tl.where(l == 0, tl.where(j[None,:] == k, 1.0, 0.0), dv)
                acc += v[:,None]*dv
        if PACK:
            m = tl.abs(j-l)
            s = tl.where(j>l, 1, 0)
            count = tl.where(m==0, 1, 2)
            width = C*(L-m+1)
            base = tl.where(m==0, 0, C*((2*m-1)*(L+1)-m*(m-1)))
            pos = base[None,:]*B+(b[:,None]*count[None,:]+s[None,:])*width[None,:]+(l-m[None,:])*C+c[:,None]
        else:
            pos = b[:,None]*dim+C*l*l+c[:,None]*d+j[None,:]
        tl.store(Y+pos, tl.where(active[:,None], acc, 0), (b[:,None]<B)&(j[None,:]<d))


    @tr.jit(do_not_specialize=["B"])
    def _natural_rotation_specialized(X,D,GATE,MASK,Y,B,C:tl.constexpr,L:tl.constexpr,
                             PACK:tl.constexpr,GATED:tl.constexpr,BC:tl.constexpr,BJ:tl.constexpr):
        # One launch, separately optimized address arithmetic for each degree.
        for degree in tl.static_range(L+1):
            if tl.program_id(1) == degree:
                _natural_rotation(X,D,GATE,MASK,Y,B,C,L,PACK,GATED,BC,BJ,degree)


    @tr.jit(do_not_specialize=["B"])
    def _natural_rotation_dot_degree(X,D,GATE,MASK,Y,B,C:tl.constexpr,L:tl.constexpr,
                             PACK:tl.constexpr,GATED:tl.constexpr,BC:tl.constexpr,
                             DEGREE:tl.constexpr,BD:tl.constexpr):
        l:tl.constexpr = DEGREE
        d:tl.constexpr = 2*l+1
        ntiles:tl.constexpr = (C+BC-1)//BC
        b = tl.program_id(0)//ntiles
        c = (tl.program_id(0)%ntiles)*BC+tl.arange(0,BC)
        dim:tl.constexpr = C*(L+1)*(L+1)
        dtotal:tl.constexpr = (L+1)*(4*(L+1)*(L+1)-1)//3
        active = tl.load(MASK+b)
        if l == 0:
            world = b*dim+c
            packed = b*C*(L+1)+c
            v = tl.load(X+tl.where(PACK,world,packed),(c<C)&active,other=0)
            if GATED:
                v *= tl.load(GATE+b*(C*(L+1)*(L+2)//2)+c,(c<C)&active,other=0)
            tl.store(Y+tl.where(PACK,packed,world),tl.where(active,v,0),c<C)
        else:
            j = tl.arange(0,BD)
            m = tl.abs(j-l)
            sign = tl.where(j>l,1,0)
            count = tl.where(m==0,1,2)
            width = C*(L-m+1)
            base = tl.where(m==0,0,C*((2*m-1)*(L+1)-m*(m-1)))
            packed = base[:,None]*B+(b*count[:,None]+sign[:,None])*width[:,None]+(l-m[:,None])*C+c[None,:]
            world = b*dim+C*l*l+c[None,:]*d+j[:,None]
            doff:tl.constexpr = l*(4*l*l-1)//3
            # The direction matrix is loaded once per channel tile, instead of
            # redundantly for every channel. IEEE dot uses FP32 arithmetic.
            if PACK:
                a = tl.load(D+b*dtotal+doff+j[None,:]*d+j[:,None],
                            (j[:,None]<d)&(j[None,:]<d)&active,other=0)
                v = tl.load(X+world,(j[:,None]<d)&(c[None,:]<C)&active,other=0)
                result = tl.dot(a,v,input_precision="ieee")
                tl.store(Y+packed,tl.where(active,result,0),(j[:,None]<d)&(c[None,:]<C))
            else:
                v = tl.load(X+packed,(j[:,None]<d)&(c[None,:]<C)&active,other=0)
                if GATED:
                    goff = C*m*(2*L+3-m)//2
                    g = tl.load(GATE+b*(C*(L+1)*(L+2)//2)+goff[:,None]+(l-m[:,None])*C+c[None,:],
                                (j[:,None]<d)&(c[None,:]<C)&active,other=0)
                    v *= g
                a = tl.load(D+b*dtotal+doff+j[None,:]*d+j[:,None],
                            (j[:,None]<d)&(j[None,:]<d)&active,other=0)
                result = tl.dot(tl.trans(v),a,input_precision="ieee")
                tl.store(Y+tl.trans(world),tl.where(active,result,0),(c[:,None]<C)&(j[None,:]<d))


    @tr.jit(do_not_specialize=["B"])
    def _natural_rotation_dot(X,D,GATE,MASK,Y,B,C:tl.constexpr,L:tl.constexpr,
                      PACK:tl.constexpr,GATED:tl.constexpr,BC:tl.constexpr,BJ:tl.constexpr):
        for degree in tl.static_range(L+1):
            if tl.program_id(1)==degree:
                # Dot dimensions have a minimum of 16; lmax=8 needs 32.
                if degree <= 7:
                    _natural_rotation_dot_degree(X,D,GATE,MASK,Y,B,C,L,PACK,GATED,BC,degree,16)
                else:
                    _natural_rotation_dot_degree(X,D,GATE,MASK,Y,B,C,L,PACK,GATED,BC,degree,BJ)


    @tr.jit(do_not_specialize=["B"])
    def _natural_gate_grad(R, Y, G, MASK, META, DY, DG, B,
                   W: tl.constexpr, BLOCK: tl.constexpr):
        e = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
        b, w = e//W, e%W
        active = tl.load(MASK+b, e<B*W, other=0)
        base = tl.load(META+4*w, e<B*W, other=0)
        width = tl.load(META+4*w+1, e<B*W, other=1)
        count = tl.load(META+4*w+2, e<B*W, other=1)
        col = tl.load(META+4*w+3, e<B*W, other=0)
        pos = base*B+b*count*width+col
        r0 = tl.load(R+pos, e<B*W, other=0)
        r1 = tl.load(R+pos+width, (e<B*W)&(count==2), other=0)
        y0 = tl.load(Y+pos, (e<B*W)&active, other=0)
        y1 = tl.load(Y+pos+width, (e<B*W)&(count==2)&active, other=0)
        g = tl.load(G+e, (e<B*W)&active, other=0)
        tl.store(DY+pos, tl.where(active, r0*g, 0), e<B*W)
        tl.store(DY+pos+width, tl.where(active, r1*g, 0), (e<B*W)&(count==2))
        tl.store(DG+e, tl.where(active, r0*y0+r1*y1, 0), e<B*W)


    def _natural_rotation(x, geometry, mask, c, l, pack, gate=None):
        b = mask.numel()
        y = torch.empty((b*c*(l+1)**2,), device=x.device, dtype=x.dtype)
        if _natural_ROTATION_VARIANT == "legacy":
            kernel,bc,grid,fusion = _natural_rotation,16,(tr.cdiv(b*c,16),l+1),False
        elif l>=4 and (pack or c>=64):
            bc = min(128,max(16,tr.next_power_of_2(c)))
            kernel,grid,fusion = _natural_rotation_dot,(b*tr.cdiv(c,bc),l+1),True
        else:
            bc = 64 if c>=64 else 32 if c>=16 and l>=4 else 16
            kernel,grid,fusion = _natural_rotation_specialized,(tr.cdiv(b*c,bc),l+1),True
        kernel[grid](
            x, geometry, gate if gate is not None else x, mask, y,
            b, c, l, pack, gate is not None, bc, tr.next_power_of_2(2*l+1),
            num_warps=4, enable_fp_fusion=fusion)
        return y if pack else y.view(b, c*(l+1)**2)


    class _natural_RotatePack(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, geometry, mask, c, l):
            ctx.save_for_backward(geometry, mask)
            ctx.c, ctx.l = c, l
            return _natural_rotation(x, geometry, mask, c, l, True)

        @staticmethod
        @once_differentiable
        def backward(ctx, grad):
            geometry, mask = ctx.saved_tensors
            return _natural_rotation(grad.contiguous(), geometry, mask, ctx.c, ctx.l, False), None, None, None, None


    class _natural_GateUnpack(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y, gate, geometry, mask, meta, c, l):
            ctx.save_for_backward(y, gate, geometry, mask, meta)
            ctx.c, ctx.l = c, l
            return _natural_rotation(y, geometry, mask, c, l, False, gate)

        @staticmethod
        @once_differentiable
        def backward(ctx, grad):
            y, gate, geometry, mask, meta = ctx.saved_tensors
            r = _natural_rotation(grad.contiguous(), geometry, mask, ctx.c, ctx.l, True)
            dy, dg = torch.empty_like(y), torch.empty_like(gate)
            b, w = gate.shape
            _natural_gate_grad[(tr.cdiv(b*w, 256),)](r, y, gate, mask, meta, dy, dg, b, w, 256,
                                           enable_fp_fusion=False)
            return dy, dg, None, None, None, None, None


    _natural_rotate_pack = _natural_RotatePack.apply
    _natural_gate_rotate_unpack = _natural_GateUnpack.apply

    @tr.jit(do_not_specialize=["B"])
    def _general_degree(X,D,G,BIAS,MASK,LAYOUT,BASES,Y,B,DIM:tl.constexpr,
                DS:tl.constexpr,W:tl.constexpr,C,START,NT:tl.constexpr,
                L:tl.constexpr,PACK:tl.constexpr,GATED:tl.constexpr,
                BC:tl.constexpr,BD:tl.constexpr):
        d:tl.constexpr=2*L+1
        b=tl.program_id(0)//NT
        c=tl.program_id(0)%NT*BC+tl.arange(0,BC)
        if b<B:
            active=tl.load(MASK+b)
            j=tl.arange(0,BD)
            worldbase=tl.load(BASES+START+c,c<C,other=0)
            world=worldbase[None,:]+j[:,None]
            valid=(j[:,None]<d)&(c[None,:]<C)
            base=tl.load(LAYOUT+6*world,valid,other=-1)
            stride=tl.load(LAYOUT+6*world+1,valid,other=1)
            col=tl.load(LAYOUT+6*world+2,valid,other=0)
            sign=tl.load(LAYOUT+6*world+3,valid,other=1)
            pos=B*base+b*stride+col
            if PACK:
                v=tl.load(X+b*DIM+world,valid&active,other=0)
            else:
                v=tl.load(X+pos,valid&(base>=0)&active,other=0)*sign
                if GATED:
                    gi=tl.load(LAYOUT+6*world+4,valid,other=-1)
                    bi=tl.load(LAYOUT+6*world+5,valid,other=-1)
                    bias=tl.load(BIAS+bi,valid&(bi>=0),other=0)
                    gate=tl.load(G+b*W+gi,valid&(gi>=0)&active,other=0)
                    v=(v+bias)*gate
            if L==0:
                result=v
            else:
                doff:tl.constexpr=L*(4*L*L-1)//3
                a=tl.load(D+b*DS+doff+j[None,:]*d+j[:,None],
                          (j[:,None]<d)&(j[None,:]<d)&active,other=0)
                if PACK:
                    result=tl.dot(a,v,input_precision="ieee")
                else:
                    result=tl.trans(tl.dot(tl.trans(v),a,input_precision="ieee"))
            if PACK:
                tl.store(Y+pos,tl.where(active,result*sign,0),valid&(base>=0))
            else:
                tl.store(Y+b*DIM+world,tl.where(active,result,0),valid)


    @tr.jit(do_not_specialize=["B"])
    def _general_rotate(X,D,G,BIAS,MASK,LAYOUT,BASES,DEGREES,Y,B,DIM:tl.constexpr,
                DS:tl.constexpr,W:tl.constexpr,LMAX:tl.constexpr,NT:tl.constexpr,
                PACK:tl.constexpr,GATED:tl.constexpr,BC:tl.constexpr,BJ:tl.constexpr):
        for l in tl.static_range(LMAX+1):
            if tl.program_id(1)==l:
                c=tl.load(DEGREES+2*l);start=tl.load(DEGREES+2*l+1)
                if c>0:
                    if l<=7:
                        _general_degree(X,D,G,BIAS,MASK,LAYOUT,BASES,Y,B,DIM,DS,W,c,start,NT,l,PACK,GATED,BC,16)
                    else:
                        _general_degree(X,D,G,BIAS,MASK,LAYOUT,BASES,Y,B,DIM,DS,W,c,start,NT,l,PACK,GATED,BC,BJ)


    def _general_rotation(x,geometry,mask,plan,side,pack,gate=None,bias=None):
        b=mask.numel()
        counts=getattr(plan,f'counts{side}')
        bc=min(64,max(16,tr.next_power_of_2(max(counts))))
        y=x.new_empty(b*(plan.sizes[side] if pack else plan.dims[side]))
        _general_rotate[(b*max(tr.cdiv(c,bc) for c in counts),len(counts))](
            x,geometry,gate if gate is not None else x,bias if bias is not None else x,mask,
            getattr(plan,f'layout{side}'),getattr(plan,f'bases{side}'),getattr(plan,f'degrees{side}'),y,b,plan.dims[side],
            geometry.shape[1],plan.w,len(counts)-1,max(tr.cdiv(c,bc) for c in counts),pack,gate is not None,bc,max(16,tr.next_power_of_2(2*len(counts)-1)),
            num_warps=4,enable_fp_fusion=True)
        return y if pack else y.view(b,plan.dims[side])


    @tr.jit
    def _general_weights(PTRS,META,FACTOR,X,Y,BACK:tl.constexpr,BLOCK:tl.constexpr):
        m=tl.program_id(1)
        ni=tl.load(META+3*m); no=tl.load(META+3*m+1); off=tl.load(META+3*m+2)
        k=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        i=k%ni; j=k//ni
        raw=tl.where(m==0,i*no+j,k)
        f=tl.load(FACTOR+off+k,k<ni*no,other=0).to(tl.float32)
        if BACK:
            v=tl.load(X+off+k,k<ni*no,other=0)
            tl.store(Y+off+raw,tl.where(f==0,0,v*f),k<ni*no)
        else:
            p=tl.cast(tl.load(PTRS+m),tl.pointer_type(tl.float32))
            v=tl.load(p+raw,k<ni*no,other=0)
            tl.store(Y+off+k,tl.where(f==0,0,v*f),k<ni*no)


    def _general_pack_weights(plan,weights):
        y=weights[0].new_empty(plan.weight_size)
        _general_weights[(tr.cdiv(plan.max_weight_size,256),len(weights))](
            plan.pointers(weights),plan.weight_meta,plan.factor,y,y,False,256)
        return y


    def _general_unpack_weight_grads(plan,dw):
        y=torch.empty_like(dw)
        _general_weights[(tr.cdiv(plan.max_weight_size,256),len(plan.groups))](
            dw,plan.weight_meta,plan.factor,dw,y,True,256)
        return y


    @tr.jit(do_not_specialize=["B"])
    def _general_gate_grad(R,Y,G,BIAS,MASK,META,DY,DG,B,W:tl.constexpr,BLOCK:tl.constexpr):
        e=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        b=e//W; w=e%W
        active=tl.load(MASK+b,e<B*W,other=0)
        base=tl.load(META+5*w,e<B*W,other=0)
        width=tl.load(META+5*w+1,e<B*W,other=1)
        count=tl.load(META+5*w+2,e<B*W,other=1)
        col=tl.load(META+5*w+3,e<B*W,other=0)
        bi=tl.load(META+5*w+4,e<B*W,other=-1)
        pos=B*base+b*count*width+col
        r0=tl.load(R+pos,(e<B*W)&active,other=0)
        r1=tl.load(R+pos+width,(e<B*W)&(count==2)&active,other=0)
        bias=tl.load(BIAS+bi,(e<B*W)&(bi>=0),other=0)
        y0=tl.load(Y+pos,(e<B*W)&active,other=0)+bias
        y1=tl.load(Y+pos+width,(e<B*W)&(count==2)&active,other=0)
        g=tl.load(G+e,(e<B*W)&active,other=0)
        tl.store(DY+pos,r0*g,e<B*W)
        tl.store(DY+pos+width,r1*g,(e<B*W)&(count==2))
        tl.store(DG+e,tl.where(active,r0*y0+r1*y1,0),e<B*W)


    @tr.jit(do_not_specialize=["B"])
    def _general_bias_grad(DY,META,DB,B,BLOCK:tl.constexpr):
        i=tl.program_id(0)
        base=tl.load(META+3*i);width=tl.load(META+3*i+1);col=tl.load(META+3*i+2)
        b=tl.arange(0,BLOCK)
        acc=tl.full((BLOCK,),0,tl.float32)
        for start in range(tl.cdiv(B,BLOCK)):
            bb=start*BLOCK+b
            acc+=tl.load(DY+B*base+bb*width+col,bb<B,other=0)
        tl.store(DB+i,tl.sum(acc,0))


    class _general_RotatePack(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,geometry,mask,plan):
            ctx.save_for_backward(geometry,mask);ctx.plan=plan
            return _general_rotation(x,geometry,mask,plan,0,True)

        @staticmethod
        @once_differentiable
        def backward(ctx,grad):
            geometry,mask=ctx.saved_tensors
            return _general_rotation(grad.contiguous(),geometry,mask,ctx.plan,0,False),None,None,None


    class _general_GateUnpack(torch.autograd.Function):
        @staticmethod
        def forward(ctx,y,gate,bias,geometry,mask,plan):
            ctx.save_for_backward(y,gate,bias,geometry,mask);ctx.plan=plan
            return _general_rotation(y,geometry,mask,plan,1,False,gate,bias)

        @staticmethod
        @once_differentiable
        def backward(ctx,grad):
            y,gate,bias,geometry,mask=ctx.saved_tensors;plan=ctx.plan
            r=_general_rotation(grad.contiguous(),geometry,mask,plan,1,True)
            dy,dg,db=torch.empty_like(y),torch.empty_like(gate),torch.empty_like(bias)
            b,w=gate.shape
            _general_gate_grad[(tr.cdiv(b*w,256),)](r,y,gate,bias,mask,plan.gate_meta,dy,dg,b,w,256,enable_fp_fusion=False)
            if bias.numel():
                _general_bias_grad[(bias.numel(),)](dy,plan.bias_meta,db,b,256)
            return dy,dg,db,None,None,None


    _general_rotate_pack=_general_RotatePack.apply
    _general_gate_rotate_unpack=_general_GateUnpack.apply


# Public optimized operator.


class PreparedWigner(list):
    """Immutable reusable geometry; construct outside timing, as with Wigner D.

    The original matrices retain their layout for baseline operators. A packed
    copy is charged to every method's common geometry/memory baseline.
    """
    def __init__(self, matrices):
        super().__init__(matrices)
        self.packed = torch.cat([d.flatten(1) for d in matrices], dim=1).contiguous()


class OptimizedTO2Linear(TO2Linear):
    def __init__(self, *args, backend="auto", linear_backend="custom", **kwargs):
        super().__init__(*args, **kwargs)
        if not self.shared_weights:
            raise NotImplementedError("Same supported branch as the reference: shared_weights=True")
        if backend not in ("auto", "torch", "triton"):
            raise ValueError(backend)
        self.backend = backend
        if linear_backend not in ("custom", "autograd"):
            raise ValueError(linear_backend)
        self.linear_backend = linear_backend
        self._refresh_layout()
        self.register_load_state_dict_post_hook(self._loaded_state)

    @classmethod
    def from_reference(cls, reference, backend="auto"):
        """Replace an existing TO2Linear while retaining weights and mode."""
        with torch.random.fork_rng(devices=[]):
            result = cls(reference.irreps_in, reference.irreps_out, None,
                         reference.shared_weights,
                         irreps_sh=reference.irreps_sh, intdtype=reference.intdtype,
                         floatdtype=next(reference.parameters()).dtype, backend=backend)
        first = next(reference.parameters())
        result.to(device=first.device, dtype=first.dtype)
        result.load_state_dict(reference.state_dict())
        result.train(reference.training)
        for name, parameter in result.named_parameters():
            parameter.requires_grad_(dict(reference.named_parameters())[name].requires_grad)
        return result

    def _refresh_layout(self):
        # All these decisions depend on representation buffers, never on data
        # or trainable parameter values. Rebuild after loading a state dict.
        self._m0_unmasked = not bool(self.to2_m_fcs[0].weight_mask.any())
        self._identity = [False] + [bool((m.cg_idx == 1).all()) for m in self.to2_m_fcs[1:]]
        for i, m in enumerate(self.to2_m_fcs[1:], 1):
            # Identity sectors need no extra quadratic masks.
            even = m.cg_idx.new_empty(0, dtype=torch.bool) if self._identity[i] else (m.cg_idx == 1).T.contiguous()
            odd = m.cg_idx.new_empty(0, dtype=torch.bool) if self._identity[i] else (m.cg_idx == 2).T.contiguous()
            m.register_buffer("_even", even, persistent=False)
            m.register_buffer("_odd", odd, persistent=False)
        order = torch.cat([m.output_m_idx for m in self.to2_m_fcs])
        self.register_buffer("_unpermute", order.argsort(), persistent=False)
        fc = self.to2_m_fcs[0]
        self._bias_prefix = torch.equal(fc.bias_idx.cpu(), torch.arange(fc.bias.numel()))
        self._input_slices = self._degree_slices(self.irreps_in)
        self._output_slices = self._degree_slices(self.irreps_out)
        c = self.irreps_in[0].mul
        natural = lambda irs: all(mul == c and ir.l == l and ir.p == (-1)**l and ir.t_p == 1
                                  for l, (mul, ir) in enumerate(irs))
        self._fused_layout = (natural(self.irreps_in) and natural(self.irreps_out)
                              and self.irreps_in.lmax == self.irreps_out.lmax
                              and self._m0_unmasked and all(self._identity[1:]) and self._bias_prefix)
        self.channels = c
        # Each radial channel maps to one or two entries in packed m layout.
        meta = []
        base = 0
        for m, fc in enumerate(self.to2_m_fcs):
            count = 1 if m == 0 else 2
            meta.extend((base, fc.c_out, count, j) for j in range(fc.c_out))
            base += count * fc.c_out
        self.register_buffer("_gate_meta", torch.tensor(meta, dtype=torch.int32, device=fc.weight.device), persistent=False)
        # Nonpersistent tables keep state_dict compatible with the reference.
        # The existing natural-parity path remains its own tuned specialization.
        self._general_layout = None
        if not self._fused_layout:
            try:
                self._general_layout = GeneralLayout(self)
            except AssertionError:
                # Nonstandard user-supplied CG buffers retain torch semantics.
                pass
        self.last_backend = None

    @staticmethod
    def _degree_slices(irreps):
        result = []
        for l in range(irreps.lmax + 1):
            spans = [s for (_, ir), s in zip(irreps, irreps.slices()) if ir.l == l]
            result.append(slice(spans[0].start, spans[-1].stop)
                          if spans and sum(s.stop-s.start for s in spans) == spans[-1].stop-spans[0].start else None)
        return result

    @staticmethod
    def _loaded_state(module, incompatible_keys):
        # Parent-model checkpoint loading must also refresh derived metadata.
        module._refresh_layout()

    def _bias(self):
        fc = self.to2_m_fcs[0]
        if self._bias_prefix:
            return F.pad(fc.bias, (0, fc.c_out-fc.bias.numel()))
        return fc.bias.new_zeros(fc.c_out).index_add(0, fc.bias_idx, fc.bias)

    def _linear(self, x, m):
        fc = self.to2_m_fcs[m]
        if m == 0:
            w = fc.weight if self._m0_unmasked else fc.weight.masked_fill(fc.weight_mask, 0)
            return F.linear(x, w.T, self._bias())
        if self._identity[m]:
            return F.linear(x, fc.weight)
        # x has (..., 2, Cin), J = [[0,1],[-1,0]].
        even = F.linear(x, fc.weight * fc._even)
        odd = F.linear(x, fc.weight * fc._odd)
        return even + torch.stack((-odd[..., 1, :], odd[..., 0, :]), dim=-2)

    def _rotate_torch(self, x, wigner, inverse=False):
        irreps = self.irreps_out if inverse else self.irreps_in
        slices = self._output_slices if inverse else self._input_slices
        chunks = []
        for l, span in enumerate(slices):
            idx = getattr(self, f"{'output' if inverse else 'input'}_{l}_idx")
            z = x[:, span] if span is not None else x.index_select(1, idx)
            if l:
                d = wigner[l].transpose(1, 2) if inverse else wigner[l]
                z = torch.bmm(z.reshape(x.shape[0], -1, 2*l+1), d).flatten(1)
            chunks.append(z)
        ordered = torch.cat(chunks, dim=1)
        # Unusual irreps ordering remains supported by a single permutation.
        if any(ir.l > other.l for (_, ir), (_, other) in zip(irreps, irreps[1:])):
            idx = torch.cat([getattr(self, f"{'output' if inverse else 'input'}_{l}_idx") for l in range(irreps.lmax+1)])
            ordered = ordered.index_select(1, idx.argsort())
        return ordered

    def _torch_forward(self, x, weight, wigner, mask_edge):
        rotated = self._rotate_torch(x, wigner)
        results = []
        for m, fc in enumerate(self.to2_m_fcs):
            z = rotated.index_select(1, fc.input_m_idx)
            g = weight[:, self.weights_slices[m]:self.weights_slices[m+1]]
            if m:
                z = z.reshape(x.shape[0], fc.c_in, 2).transpose(1, 2)
                y = (self._linear(z, m) * g.unsqueeze(1)).transpose(1, 2).flatten(1)
            else:
                y = self._linear(z, m) * g
            results.append(y)
        combined = torch.cat(results, dim=1)
        if self._unpermute.numel() == self.irreps_out.dim:
            out = combined.index_select(1, self._unpermute)
        else:
            order = torch.cat([fc.output_m_idx for fc in self.to2_m_fcs])
            out = x.new_zeros(x.shape[0], self.irreps_out.dim).index_copy(1, order, combined)
        out = torch.where(mask_edge[:, None], out, 0)
        return self._rotate_torch(out, wigner, inverse=True)

    def forward(self, x, length_emb=None, wigner_D=None, edge_sh=None,
                mask_edge=None, mask_sc=None, weight=None):
        if mask_edge is None:
            raise ValueError("mask_edge is required, as in the reference")
        supported = (self.backend != "torch" and _TRITON_AVAILABLE and x.is_cuda
                 and x.dtype == torch.float32 and x.is_contiguous()
                 and weight.is_contiguous() and mask_edge.is_contiguous()
                 and weight.dtype == x.dtype and weight.device == x.device
                 and all(d.dtype==x.dtype and d.device==x.device for d in wigner_D)
                 and not any(d.requires_grad for d in wigner_D)
                 and all(p.is_contiguous() and p.device==x.device and p.dtype==x.dtype for p in self.parameters())
                 and x.shape[0]>0)
        fused = supported and self._fused_layout
        general = supported and self._general_layout is not None
        if self.backend == "triton" and not (fused or general):
            raise ValueError("Triton needs contiguous CUDA FP32 and fixed geometry; use torch for other cases")
        if not (fused or general):
            self.last_backend = "torch"
            return self._torch_forward(x, weight, wigner_D, mask_edge)
        if general:
            self.last_backend = "triton_general"
            geometry = wigner_D.packed if isinstance(wigner_D, PreparedWigner) else PreparedWigner(wigner_D).packed
            packed = _general_rotate_pack(x,geometry,mask_edge,self._general_layout)
            linear = general_linear(packed,self._general_layout,x.shape[0],*[fc.weight for fc in self.to2_m_fcs])
            return _general_gate_rotate_unpack(linear,weight,self.to2_m_fcs[0].bias,geometry,mask_edge,self._general_layout)
        self.last_backend = "triton_natural"
        return self._fused_forward(x, weight, wigner_D, mask_edge)

    def _fused_forward(self, x, weight, wigner_D, mask_edge):
        # The specialized kernel uses a stride determined by its own lmax.
        # Extra matrices are permitted, but must not change that stride.
        matrices = wigner_D if len(wigner_D)==self.m_max+1 else wigner_D[:self.m_max+1]
        geometry = matrices.packed if isinstance(matrices, PreparedWigner) else PreparedWigner(matrices).packed
        packed = _natural_rotate_pack(x, geometry, mask_edge, self.channels, self.m_max)
        b = x.shape[0]
        if self.linear_backend == "custom":
            linear = structured_linear(packed,self.to2_m_fcs[0].bias,b,self.channels,
                                       self.m_max,*[fc.weight for fc in self.to2_m_fcs])
            return _natural_gate_rotate_unpack(linear,weight,geometry,mask_edge,
                                      self._gate_meta,self.channels,self.m_max)
        offset = 0
        ys = []
        for m, fc in enumerate(self.to2_m_fcs):
            rows = b if m == 0 else b*2
            size = rows * fc.c_in
            z = packed[offset:offset+size].view(rows, fc.c_in)
            ys.append(self._linear(z, m).flatten())
            offset += size
        linear = torch.cat(ys)
        return _natural_gate_rotate_unpack(linear, weight, geometry, mask_edge,
                                  self._gate_meta, self.channels, self.m_max)
