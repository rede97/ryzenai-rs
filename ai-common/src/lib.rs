pub mod runtime;
#[macro_export]
macro_rules! measure_time {
    ($block:block) => {{
        let start = std::time::Instant::now();
        let result = $block;
        let duration = start.elapsed();
        (result, duration)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measure_time() {
        let (_, duration) = measure_time!({
            let mut sum = 0;
            for i in 0..1000 {
                sum += i;
            }
            assert_eq!(sum, 499500);
        });
        println!("Duration: {:?}", duration);
    }
}
