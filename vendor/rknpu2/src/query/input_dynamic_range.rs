use {
    crate::{
        query::QueryWithInput,
        tensor::{TensorFormatKind, TensorFormatKind::UNDEFINED},
    },
    rknpu2_sys::rknn_input_range,
    std::ffi::CStr,
};

/// Query supported dynamic shape profiles for a specific input tensor.
pub struct InputDynamicRange {
    pub(crate) inner: rknn_input_range,
}

impl QueryWithInput for InputDynamicRange {
    const QUERY_TYPE: rknpu2_sys::_rknn_query_cmd::Type =
        rknpu2_sys::_rknn_query_cmd::RKNN_QUERY_INPUT_DYNAMIC_RANGE;

    type Input = u32;
    type Output = rknn_input_range;

    fn prepare(input: Self::Input, output: &mut Self::Output) {
        output.index = input;
    }
}

impl From<rknn_input_range> for InputDynamicRange {
    fn from(inner: rknn_input_range) -> Self {
        Self { inner }
    }
}

impl InputDynamicRange {
    pub fn index(&self) -> u32 {
        self.inner.index
    }

    pub fn name(&self) -> String {
        let cstr = unsafe { CStr::from_ptr(self.inner.name.as_ptr()) };
        cstr.to_string_lossy().into_owned()
    }

    pub fn format(&self) -> TensorFormatKind {
        self.inner.fmt.into()
    }

    pub fn n_dims(&self) -> usize {
        self.inner.n_dims as usize
    }

    pub fn shape_number(&self) -> usize {
        self.inner.shape_number as usize
    }

    pub fn shape_at(&self, idx: usize) -> Option<Vec<u32>> {
        if idx >= self.shape_number() {
            return None;
        }
        let nd = self.n_dims();
        if nd == 0 {
            return Some(Vec::new());
        }
        Some(self.inner.dyn_range[idx][..nd].to_vec())
    }

    pub fn shapes(&self) -> Vec<Vec<u32>> {
        (0..self.shape_number())
            .filter_map(|i| self.shape_at(i))
            .collect()
    }

    pub fn is_valid_dynamic(&self) -> bool {
        self.shape_number() > 0 && self.n_dims() > 0 && !matches!(self.format(), UNDEFINED(_))
    }
}

