package lyra;

public class versionMismatchException extends IllegalStateException {
    public versionMismatchException (String message) {
        super(message);
    }
}
